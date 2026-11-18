import importlib.util
import json
import pathlib
import struct
import tempfile
import unittest

MODULE = pathlib.Path(__file__).resolve().parents[1] / 'tools' / 'lowram_tandem_run.py'
spec = importlib.util.spec_from_file_location('lowram_launcher', MODULE)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def string(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


class GGUFMetadataTests(unittest.TestCase):
    def test_expert_keys_after_tokenizer_are_read(self):
        entries = [
            string('general.architecture') + struct.pack('<I', 8) + string('deepseek4'),
            string('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, 2) + string('hello') + string('world'),
            string('deepseek4.expert_count') + struct.pack('<II', 4, 256),
            string('deepseek4.expert_used_count') + struct.pack('<II', 4, 6),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'model.gguf'
            path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, len(entries)) + b''.join(entries))
            info = launcher.gguf_info(str(path))
        self.assertEqual(info['arch'], 'deepseek4')
        self.assertEqual(info['experts'], 256)
        self.assertEqual(info['experts_used'], 6)

    def test_incomplete_metadata_remains_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'model.gguf'
            path.write_bytes(b'not a GGUF')
            self.assertEqual(launcher.gguf_info(str(path)), {})

    def test_unknown_expert_count_uses_moe_flags_and_requested_layers(self):
        hw = json.dumps(dict(gpu='RTX 4060', vram_gib=8, cores=8, ram_gib=40, pcie_gen=3))
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'incomplete.gguf'
            path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 0, 1) +
                             string('general.architecture') + struct.pack('<I', 8) + string('deepseek4'))
            args = launcher.parse_args(['--model', str(path), '--dry-run', '--fake-hw', hw, '--gpu-layers', '10'])
            machine, model = launcher.make_plan(args)
            model.size = 92 * launcher.GiB
            plan = launcher.choose_flags(machine, model, args)
        self.assertIn('-cmoe', plan.args)
        self.assertEqual(plan.args[plan.args.index('-ngl') + 1], '10')
        self.assertTrue(any('expert count is unknown' in warning for warning in plan.warnings))


if __name__ == '__main__':
    unittest.main()
