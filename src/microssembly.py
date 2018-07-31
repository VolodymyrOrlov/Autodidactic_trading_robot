from utils import FloatConvert32
import numpy as np

class Microssembly:

    def __init__(self, in_memory_length = 4, out_memory_length = 1, registers_length = 1, int_length = 8, trace=True):
        self.in_memory_length = in_memory_length
        self.out_memory_length = out_memory_length
        self.registers_length = registers_length
        self.command_length = 4
        self.int_length = int_length
        self.trace = trace
        self.reset()

    def reset(self):
        self.ignore = False
        self.registry = np.zeros(2**self.registers_length)
        self.in_memory = np.zeros(2**self.in_memory_length)
        self.out_memory = np.zeros(2**self.out_memory_length)

    def load_data(self, data):
        assert isinstance(data, list), 'data should be an instance of list'
        assert len(data) <= 2**self.in_memory_length, 'data size should not exceed [{}]'.format(self.in_memory_length)
        np.copyto(self.in_memory[:len(data)], data)

    def run(self,  code):
        assert isinstance(code, str), 'code should be an instance of string'
        trace = []
        while len(code) >= self.command_length:
            try:
                code = self._parse_cmd(code[:self.command_length], code[self.command_length:],
                                          trace if self.trace else None)
            except:
                code = ''
        return trace

    def _read_reg_pos(self, body):
        return int(body[:self.registers_length], 2), body[self.registers_length:]

    def _read_int(self, body):
        return int(body[:self.int_length], 2), body[self.int_length:]

    def _read_in_mem_pos(self, body):
        return int(body[:self.in_memory_length], 2), body[self.in_memory_length:]

    def _read_out_mem_pos(self, body):
        return int(body[:self.out_memory_length], 2), body[self.out_memory_length:]

    def _read_const(self, body):
        return FloatConvert32.bin_to_float(body[:self.float_length]), body[self.float_length:]

    def _parse_cmd(self, binary_cmd, body, trace):
        if binary_cmd == '0000': # ignore-if
            src, body = self._read_reg_pos(body)
            if trace is not None and not self.ignore:
                trace.append('ignore-if [{}]'.format(src))
            if bool(self.registry[src]):
                self.ignore = True
            return body
        if binary_cmd == '0001': # load
            src, body = self._read_in_mem_pos(body)
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = self.in_memory[src]
            if trace is not None and not self.ignore:
                trace.append('load [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '0010': # set
            dst, body = self._read_reg_pos(body)
            value, body = self._read_int(body)
            if not self.ignore:
                self.registry[dst] = value
            if trace is not None:
                trace.append('set [{}] [{}]'.format(value, dst))
            return body
        if binary_cmd == '0011': # add
            src, body = self._read_reg_pos(body)
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = self.registry[src] + self.registry[dst]
            if trace is not None and not self.ignore:
                trace.append('add [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '0100': # div
            src, body = self._read_reg_pos(body)
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = self.registry[src] / self.registry[dst]
            if trace is not None and not self.ignore:
                trace.append('div [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '0101': # unload
            src, body = self._read_reg_pos(body)
            dst, body = self._read_out_mem_pos(body)
            if not self.ignore:
                self.out_memory[dst] = self.registry[src]
            if trace is not None and not self.ignore:
                trace.append('unload [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '0110': # ignore
            if trace is not None and not self.ignore:
                trace.append('ignore')
            self.ignore = True
            return body
        if binary_cmd == '0111': # stop-ignore
            self.ignore = False
            if trace is not None:
                trace.append('stop-ignore')
            return body
        if binary_cmd == '1000': # max
            src, body = self._read_reg_pos(body)
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = max(self.registry[src], self.registry[dst])
            if trace is not None and not self.ignore:
                trace.append('max [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1001': # min
            src, body = self._read_reg_pos(body)
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = min(self.registry[src], self.registry[dst])
            if trace is not None and not self.ignore:
                trace.append('min [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1010': # inc
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = self.registry[dst] + 1
            if trace is not None and not self.ignore:
                trace.append('inc [{}]'.format(dst))
            return body
        if binary_cmd == '1011': # dec
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = self.registry[dst] - 1
            if trace is not None and not self.ignore:
                trace.append('dec [{}]'.format(dst))
            return body
        if binary_cmd == '1100': # mov
            src, body = self._read_reg_pos(body)
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = self.registry[src]
            if trace is not None and not self.ignore:
                trace.append('mov [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1101': # bin-max
            src, body = self._read_reg_pos(body)
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = 1 if self.registry[src] > self.registry[dst] else 0
            if trace is not None and not self.ignore:
                trace.append('bin-max [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1110': # bin-min
            src, body = self._read_reg_pos(body)
            dst, body = self._read_reg_pos(body)
            if not self.ignore:
                self.registry[dst] = 1 if self.registry[src] < self.registry[dst] else 0
            if trace is not None and not self.ignore:
                trace.append('bin-min [{}] [{}]'.format(src, dst))
            return body
        else:
            return body

