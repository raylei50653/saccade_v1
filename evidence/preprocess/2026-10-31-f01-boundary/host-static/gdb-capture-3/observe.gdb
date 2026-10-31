set pagination off
set confirm off
set breakpoint pending on
set print thread-events off
set debuginfod enabled off
set disable-randomization off
python
import gdb
import hashlib
import json
import os

def u64(p):
    return int.from_bytes(bytes(gdb.selected_inferior().read_memory(p, 8)), 'little')

def emit(kind, **kw):
    print('F01_GDB ' + json.dumps(dict(kind=kind, **kw), sort_keys=True))

def bt():
    print(gdb.execute('bt 9', to_string=True))

class SlotBreakpoint(gdb.Breakpoint):
    def stop(self):
        emit('private_slot_70_entry', pc=hex(int(gdb.parse_and_eval('$pc'))),
             rdi=hex(int(gdb.parse_and_eval('$rdi'))),
             rsi=hex(int(gdb.parse_and_eval('$rsi'))),
             rdx=hex(int(gdb.parse_and_eval('$rdx'))),
             owner=gdb.solib_name(int(gdb.parse_and_eval('$pc'))))
        bt()
        return False

class GetterFinish(gdb.FinishBreakpoint):
    def __init__(self, table):
        super().__init__(gdb.newest_frame(), internal=True)
        self.table = table
    def stop(self):
        values = {hex(i): hex(u64(self.table+i)) for i in range(0, 0xa0, 8)}
        emit('private_getter_return', table=hex(self.table), values=values,
             return_value=str(self.return_value), slot70_owner=gdb.solib_name(u64(self.table+0x70)))
        SlotBreakpoint('*' + hex(u64(self.table+0x70)), internal=True)
        return False

class GetterBreakpoint(gdb.Breakpoint):
    def stop(self):
        table=int(gdb.parse_and_eval('$rdi'))
        emit('private_getter_entry', table=hex(table), pc=hex(int(gdb.parse_and_eval('$pc'))))
        bt()
        GetterFinish(table)
        return False

class DispatchBreakpoint(gdb.Breakpoint):
    def stop(self):
        vptr=int(gdb.parse_and_eval('$r10'))
        target=u64(vptr+0x38)
        ti=u64(vptr-8)
        name=gdb.Value(u64(ti+8)).cast(gdb.lookup_type('char').pointer()).string()
        emit('actual_backend_dispatch', pc=hex(int(gdb.parse_and_eval('$pc'))),
             vptr=hex(vptr), target=hex(target), target_owner=gdb.solib_name(target), rtti_name=name)
        print(gdb.execute('x/i $pc',to_string=True))
        bt()
        return False

class DecodeBreakpoint(gdb.Breakpoint):
    def stop(self):
        entry=int(gdb.parse_and_eval('(void*)nvjpegDecodeBatched'))
        modules={}
        maps=open('/proc/'+str(gdb.selected_inferior().pid)+'/maps').read()
        for line in maps.splitlines():
            fields=line.split()
            if len(fields)>=6 and any(x in fields[-1] for x in ('libnvjpeg','libnvcuvid','libcuda.so','libdxcore')):
                path=fields[-1]
                if path not in modules:
                    with open(path,'rb') as f:
                        modules[path]=dict(sha256=hashlib.sha256(f.read()).hexdigest(),realpath=os.path.realpath(path))
        emit('decode_entry', entry=hex(entry), breakpoint_pc=hex(int(gdb.parse_and_eval('$pc'))), modules=modules)
        print(gdb.execute('info sharedlibrary',to_string=True))
        print(gdb.execute('info proc mappings',to_string=True))
        DispatchBreakpoint('*' + hex(entry+0x95), internal=True)
        self.enabled=False
        return False

GetterBreakpoint('__std_2U4S4U_X08', internal=True)
DecodeBreakpoint('nvjpegDecodeBatched', internal=True)
gdb.events.exited.connect(lambda event: emit('inferior_exited',exit_code=getattr(event,'exit_code',None)))
end
run
python
try:
    print('F01_GDB_INFERIOR_EXIT', gdb.parse_and_eval('$_exitcode'))
except gdb.error as e:
    print('F01_GDB_NO_EXIT_CODE', str(e))
end
