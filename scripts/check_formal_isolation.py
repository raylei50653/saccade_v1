#!/usr/bin/env python3
"""Audit a completed CMake build; requires codemodel-v2, compile DB and real deps.

Before configure: mkdir -p BUILD/.cmake/api/v1/query; touch .../codemodel-v2
Configure with CMAKE_EXPORT_COMPILE_COMMANDS=ON, then build all formal targets.
This reads existing build data. Physical research-absent configure/build is a
separate validation; this tool never renames source directories or configures.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

CPU_ROOTS = ('protocol_model', 'packet_compare', 'model_manifest', 'head_weights', 'track_offline', 'gmc_records')
CPU_OPTIONAL_ROOTS = ('saccade_gmc_g1', 'gmc_g1_tests', 'flow_sequence', 'ofa_audit')
OPTIONAL_ROOTS = ('saccade_tracking_cuda', 'saccade_flow_motion', 'saccade_tracking_session_kernels',
                  'saccade_tracking_session', 'track_sequence', 'track_stream')
RUNTIME_ROOTS = set(OPTIONAL_ROOTS)
BANNED_TARGETS = {'saccade_occlusion_experiment', 'occlusion_ablation', 'tracking_event_trace',
                  'saccade_gmc_flow', 'gmc_flow_offline'}
BANNED_FILES = {'occlusion_experiment.cpp', 'occlusion_experiment.hpp', 'occlusion_ablation.cpp',
                'tracking_event_trace.cpp', 'tracking_events.py', 'tracking_event_viewer.html',
                'gmc_flow.cpp', 'gmc_flow.hpp', 'gmc_flow_offline.cpp'}
BANNED_TEXT = re.compile(r'\b(?:TrackControlHook|AssociationCostHook|OcclusionExperiment)\b')
BANNED_SYMBOLS = re.compile(r'OcclusionExperiment|saccade::gmc_flow::|TrackControlHook|AssociationCostHook|TrackerReference::set_control_hook')


class IsolationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise IsolationError(message)


def read_json(path):
    require(path.is_file(), f'missing required build data: {path}')
    return json.loads(path.read_text())


def execute(argv):
    result = subprocess.run(argv, text=True, capture_output=True, check=False)
    require(result.returncode == 0, f'command failed: {argv}: {result.stderr.strip()}')
    return result.stdout


def model(build, source):
    reply = build / '.cmake/api/v1/reply'
    indices = sorted(reply.glob('index-*.json'), key=lambda p: p.stat().st_mtime_ns)
    require(indices, 'missing CMake File API index; request codemodel-v2 before configure')
    index = read_json(indices[-1])
    refs = [r for r in index.get('objects', []) if r.get('kind') == 'codemodel' and r.get('version', {}).get('major') == 2]
    require(len(refs) == 1, 'missing or ambiguous codemodel-v2 response')
    code = read_json(reply / refs[0]['jsonFile'])
    require(Path(code['paths']['source']).resolve() == source, 'codemodel source differs from --source')
    require(Path(code['paths']['build']).resolve() == build, 'codemodel build differs from --build')
    require(len(code['configurations']) == 1, 'only single-configuration builds are supported')
    targets = {}
    for ref in code['configurations'][0]['targets']:
        target = read_json(reply / ref['jsonFile'])
        require(target['id'] == ref['id'], 'target identity mismatch in codemodel')
        targets[target['id']] = target
    require(targets, 'empty codemodel')
    return targets


def closure(root, targets):
    found = set()
    todo = [root]
    while todo:
        ident = todo.pop()
        require(ident in targets, f'unknown target dependency: {ident}')
        if ident not in found:
            found.add(ident)
            todo.extend(d['id'] for d in targets[ident].get('dependencies', []))
    return found


def absolute(path, directory):
    p = Path(path)
    return (p if p.is_absolute() else directory / p).resolve()


def check_path(path, research, *, inspect_text=False):
    require(not path.is_relative_to(research), f'research path in formal dependency: {path}')
    require(path.name not in BANNED_FILES, f'experiment file in formal dependency: {path}')
    require(path.exists(), f'missing compiled dependency: {path}')
    if inspect_text and path.suffix in {'.cpp', '.cu', '.c', '.hpp', '.h', '.cuh', '.hh', '.hxx'}:
        require(not BANNED_TEXT.search(path.read_text(errors='replace')), f'experiment interface in formal dependency: {path}')


def object_path(entry):
    arguments = entry.get('arguments') or shlex.split(entry.get('command', ''))
    output = entry.get('output')
    if output is None:
        require('-o' in arguments, 'compile command has no object output')
        output = arguments[arguments.index('-o') + 1]
    return absolute(output, Path(entry['directory'])), arguments


def dependency_paths(obj, build):
    depfile = Path(str(obj) + '.d')
    if depfile.is_file():
        text = depfile.read_text().replace('\\\n', '')
        require(':' in text, f'invalid compiler depfile: {depfile}')
        dependencies = shlex.split(text.split(':', 1)[1])
        # GCC may append -MP phony rules. CMake's ordinary generated files do not.
        require(not any(p.endswith(':') for p in dependencies), f'unsupported multiple-rule depfile: {depfile}')
    elif (build / 'build.ninja').is_file():
        text = execute(['ninja', '-C', str(build), '-t', 'deps', str(obj.relative_to(build))])
        require(': #deps ' in text, f'missing Ninja compiler dependencies: {obj}')
        dependencies = [line.strip() for line in text.splitlines() if line.startswith('    ')]
    else:
        raise IsolationError(f'missing compiler depfile: {depfile}')
    require(dependencies, f'empty compiler dependency set: {obj}')
    return [absolute(p, build) for p in dependencies]


def audit(source, build, research, extra_roots=()):
    source, build, research = Path(source).resolve(), Path(build).resolve(), Path(research).resolve()
    targets = model(build, source)
    by_name = {t['name']: ident for ident, t in targets.items()}
    roots = list(dict.fromkeys(CPU_ROOTS + tuple(n for n in CPU_OPTIONAL_ROOTS + OPTIONAL_ROOTS if n in by_name) + tuple(extra_roots)))
    require(all(n in by_name for n in roots), f'missing required formal targets: {set(roots) - set(by_name)}')
    selected = set()
    for root in roots:
        current = closure(by_name[root], targets)
        names = {targets[i]['name'] for i in current}
        require(not names.intersection(BANNED_TARGETS), f'experiment target linked by {root}: {names.intersection(BANNED_TARGETS)}')
        if root in RUNTIME_ROOTS:
            require('saccade_tracking_reference' not in names, f'CPU tracker reference linked by CUDA runtime {root}')
            require('saccade_flow_cuda' not in names, f'standalone flow extraction linked by CUDA runtime {root}')
        selected.update(current)
    database = read_json(build / 'compile_commands.json')
    require(isinstance(database, list) and database, 'empty or invalid compile database')
    commands = {}
    for entry in database:
        obj, args = object_path(entry)
        for ident in selected:
            if f"CMakeFiles/{targets[ident]['name']}.dir/" in str(obj):
                commands.setdefault(ident, []).append((entry, obj, args))
    checked_sources, checked_dependencies, artifacts = set(), set(), set()
    for ident in sorted(selected):
        target = targets[ident]
        groups = target.get('compileGroups', [])
        for group in groups:
            for include in group.get('includes', []):
                check_path(absolute(include['path'], source), research)
        expected = {absolute(s['path'], build if s.get('isGenerated') else source)
                    for s in target.get('sources', []) if 'compileGroupIndex' in s}
        entries = commands.get(ident, [])
        actual = {absolute(e['file'], Path(e['directory'])) for e, _, _ in entries}
        require(expected == actual, f'compile database coverage mismatch for {target["name"]}: missing {expected - actual}; extra {actual - expected}')
        for entry, obj, args in entries:
            require(obj.is_file(), f'missing compiled object: {obj}')
            src = absolute(entry['file'], Path(entry['directory']))
            check_path(src, research, inspect_text=True)
            checked_sources.add(str(src))
            for arg in args:
                require(str(research) not in arg, f'research path in compile command for {src}')
            deps = dependency_paths(obj, build)
            require(src in deps, f'compiler dependency data omits source: {src}')
            for dep in deps:
                if str(dep) not in checked_dependencies:
                    check_path(dep, research, inspect_text=True)
                    checked_dependencies.add(str(dep))
        if target['type'] in {'EXECUTABLE', 'STATIC_LIBRARY', 'SHARED_LIBRARY', 'MODULE_LIBRARY'}:
            require(target.get('artifacts'), f'missing declared artifact for {target["name"]}')
            for declared in target['artifacts']:
                artifact = absolute(declared['path'], build)
                require(artifact.is_file(), f'missing enabled artifact: {artifact}')
                symbols = execute(['nm', '-C', str(artifact)])
                require(symbols.strip() and not re.search(r'(^|\n).*: no symbols\s*$', symbols), f'no usable symbols in artifact: {artifact}')
                require(not BANNED_SYMBOLS.search(symbols), f'experiment symbol in formal artifact: {artifact}')
                artifacts.add(str(artifact))
    return {'schema': 'saccade-formal-isolation/1', 'pass': True, 'source': str(source), 'build': str(build),
            'research_root': str(research), 'roots': roots, 'targets': sorted(targets[i]['name'] for i in selected),
            'compiled_sources': sorted(checked_sources), 'compiler_dependency_count': len(checked_dependencies),
            'artifacts': sorted(artifacts), 'scope': 'Configured target closure, compiled source/include dependencies and unstripped symbols; physical research-absent build is separate.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--research-root', type=Path, default=Path.home() / 'developer/ai/saccade_research')
    parser.add_argument('--target', action='append', default=[], help='Additional root; required CPU roots remain checked')
    args = parser.parse_args(argv)
    try:
        report = audit(args.source, args.build, args.research_root, args.target)
    except (IsolationError, OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({'schema': 'saccade-formal-isolation/1', 'pass': False, 'error': str(exc)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
