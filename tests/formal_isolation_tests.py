"""Contract tests for isolation failures invisible to link-symbol grepping."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_formal_isolation.py'
spec = importlib.util.spec_from_file_location('formal_isolation', SCRIPT)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.build, self.research = (self.root / n for n in ('formal', 'build', 'research'))
        for p in (self.source, self.build, self.research):
            p.mkdir()
        self.reply = self.build / '.cmake/api/v1/reply'
        self.reply.mkdir(parents=True)
        self.targets, self.commands = [], []
        for name in checker.CPU_ROOTS:
            self.add_target(name)
        self.save()
        self.run_mock = patch.object(checker.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '0001 T main\n', ''))
        self.run_mock.start()
        self.addCleanup(self.run_mock.stop)

    def add_target(self, name, *, dependencies=(), source=None):
        path = self.source / (source or f'{name}.cpp')
        path.write_text('int formal_function() { return 1; }\n')
        obj = self.build / f'CMakeFiles/{name}.dir/{path.name}.o'
        obj.parent.mkdir(parents=True)
        obj.write_bytes(b'object placeholder')
        Path(str(obj) + '.d').write_text(f'{obj}: {path}\n')
        artifact = self.build / name
        artifact.write_bytes(b'unstripped fixture placeholder')
        target = {'id': name + '::id', 'name': name, 'type': 'EXECUTABLE',
                  'dependencies': [{'id': d + '::id'} for d in dependencies],
                  'sources': [{'path': path.name, 'compileGroupIndex': 0}],
                  'compileGroups': [{}], 'artifacts': [{'path': name}]}
        self.targets.append(target)
        self.commands.append({'directory': str(self.build), 'file': str(path), 'output': str(obj),
                              'arguments': ['c++', '-o', str(obj), '-c', str(path)]})
        return target, obj, path

    def save(self):
        refs = []
        for t in self.targets:
            filename = f'target-{t["name"]}.json'
            (self.reply / filename).write_text(json.dumps(t))
            refs.append({'id': t['id'], 'name': t['name'], 'jsonFile': filename})
        (self.reply / 'codemodel.json').write_text(json.dumps({'paths': {'source': str(self.source), 'build': str(self.build)},
                                                            'configurations': [{'targets': refs}]}))
        (self.reply / 'index-test.json').write_text(json.dumps({'objects': [{'kind': 'codemodel', 'version': {'major': 2}, 'jsonFile': 'codemodel.json'}]}))
        (self.build / 'compile_commands.json').write_text(json.dumps(self.commands))

    def audit(self, roots=()):
        return checker.audit(self.source, self.build, self.research, roots)

    def test_stock_build_checks_cpu_roots_and_actual_deps(self):
        result = self.audit()
        self.assertTrue(result['pass'])
        self.assertEqual(result['roots'], list(checker.CPU_ROOTS))
        self.assertEqual(result['compiler_dependency_count'], len(checker.CPU_ROOTS))

    def test_extra_root_does_not_omit_default_cpu_roots(self):
        self.add_target('official_probe')
        self.save()
        result = self.audit(['official_probe'])
        self.assertEqual(set(result['roots']), set(checker.CPU_ROOTS) | {'official_probe'})

    def test_compiled_experiment_source_rejected(self):
        self.add_target('official_probe', source='occlusion_experiment.cpp')
        self.save()
        with self.assertRaisesRegex(checker.IsolationError, 'experiment file'):
            self.audit(['official_probe'])

    def test_hidden_inline_hook_in_depfile_header_rejected(self):
        header = self.source / 'innocent.hpp'
        header.write_text('using AssociationCostHook = int;\n')
        obj = Path(self.commands[0]['output'])
        with Path(str(obj) + '.d').open('a') as f:
            f.write(' ' + str(header))
        with self.assertRaisesRegex(checker.IsolationError, 'experiment interface'):
            self.audit()

    def test_symlink_header_to_research_rejected(self):
        actual = self.research / 'observer.hpp'
        actual.write_text('int observer;\n')
        alias = self.source / 'alias.hpp'
        alias.symlink_to(actual)
        obj = Path(self.commands[0]['output'])
        with Path(str(obj) + '.d').open('a') as f:
            f.write(' ' + str(alias))
        with self.assertRaisesRegex(checker.IsolationError, 'research path'):
            self.audit()

    def test_experiment_link_dependency_rejected(self):
        self.add_target('saccade_gmc_flow')
        self.targets[0]['dependencies'] = [{'id': 'saccade_gmc_flow::id'}]
        self.save()
        with self.assertRaisesRegex(checker.IsolationError, 'experiment target linked'):
            self.audit()

    def test_cpu_reference_is_allowed_offline_but_rejected_in_runtime(self):
        self.add_target('saccade_tracking_reference')
        self.targets[-2]['dependencies'] = [{'id': 'saccade_tracking_reference::id'}]
        self.add_target('saccade_tracking_cuda', dependencies=['saccade_tracking_reference'])
        self.save()
        with self.assertRaisesRegex(checker.IsolationError, 'CPU tracker reference linked'):
            self.audit()

    def test_official_flow_tools_allowed_and_audited(self):
        self.add_target('saccade_flow_cuda')
        self.add_target('flow_sequence', dependencies=['saccade_flow_cuda'])
        self.add_target('ofa_audit')
        self.save()
        result = self.audit()
        self.assertIn('flow_sequence', result['roots'])
        self.assertIn('ofa_audit', result['roots'])
        self.assertIn('saccade_flow_cuda', result['targets'])
        self.assertIn('gmc_records', result['roots'])

    def test_standalone_flow_library_rejected_only_in_production_runtime(self):
        self.add_target('saccade_flow_cuda')
        self.add_target('saccade_tracking_session', dependencies=['saccade_flow_cuda'])
        self.save()
        with self.assertRaisesRegex(checker.IsolationError, 'standalone flow extraction linked'):
            self.audit()

    def test_missing_enabled_artifact_rejected(self):
        (self.build / checker.CPU_ROOTS[-1]).unlink()
        with self.assertRaisesRegex(checker.IsolationError, 'missing enabled artifact'):
            self.audit()

    def test_missing_compile_database_rejected(self):
        (self.build / 'compile_commands.json').unlink()
        with self.assertRaisesRegex(checker.IsolationError, 'missing required build data'):
            self.audit()

    def test_missing_dependency_data_rejected(self):
        Path(self.commands[0]['output'] + '.d').unlink()
        with self.assertRaisesRegex(checker.IsolationError, 'missing compiler depfile'):
            self.audit()

    def test_missing_compile_entry_rejected(self):
        self.commands.pop()
        self.save()
        with self.assertRaisesRegex(checker.IsolationError, 'coverage mismatch'):
            self.audit()

    def test_symbols_are_supplementary_and_specific(self):
        with patch.object(checker.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '001 T saccade::gmc_flow::median()\n', '')):
            with self.assertRaisesRegex(checker.IsolationError, 'experiment symbol'):
                self.audit()
        with patch.object(checker.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', 'nm: no symbols')):
            with self.assertRaisesRegex(checker.IsolationError, 'no usable symbols'):
                self.audit()
        with patch.object(checker.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '001 T ordinary_Hook_documentation\n', '')):
            self.assertTrue(self.audit()['pass'])


if __name__ == '__main__':
    unittest.main()
