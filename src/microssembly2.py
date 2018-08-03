import numpy as np

class Microssembly:

    def __init__(self, architecture=8, trace=True):
        self.memory_length = architecture
        self.command_length = 4
        self.int_length = architecture
        self.trace = trace
        self.reset()

    def reset(self):
        self.memory = np.zeros(2**self.memory_length)

    def load_data(self, data):
        assert isinstance(data, list), 'data should be an instance of list'
        assert len(data) <= 2**self.memory_length, 'data size should not exceed [{}]'.format(self.memory_length)
        np.copyto(self.memory[:len(data)], data)

    def run(self,  code, cycles=100):
        assert isinstance(code, str), 'code should be an instance of string'
        trace = []
        cycle = 0
        eip = code
        while len(eip) >= self.command_length and cycle <= cycles:
            try:
                eip = self._parse_cmd(eip[:self.command_length], eip[self.command_length:], code,
                                          trace if self.trace else None)
                cycle += 1
            except:
                eip = ''
        return trace

    def _read_int(self, body):
        return int(body[:self.int_length], 2), body[self.int_length:]

    def _read_mem_pos(self, body):
        return int(body[:self.memory_length], 2), body[self.memory_length:]

    def _move_pos(self, body):
        return body[self.memory_length:]

    def _parse_cmd(self, binary_cmd, body, code, trace):
        if binary_cmd == '0010': # set
            dst, body = self._read_mem_pos(body)
            value, body = self._read_int(body)
            self.memory[dst] = value
            if trace is not None:
                trace.append('set [{}] [{}]'.format(value, dst))
            return body
        if binary_cmd == '0011': # add
            src, body = self._read_mem_pos(body)
            dst, body = self._read_mem_pos(body)
            self.memory[dst] = self.memory[src] + self.memory[dst]
            if trace is not None:
                trace.append('add [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '0100': # div
            src, body = self._read_mem_pos(body)
            dst, body = self._read_mem_pos(body)
            self.memory[dst] = self.memory[src] / self.memory[dst]
            if trace is not None:
                trace.append('div [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '0110': # jmp
            pos, _ = self._read_int(body)
            if trace is not None:
                trace.append('jmp [{}]'.format(pos))
            return code[(pos // (self.command_length + self.memory_length * 2) * (self.command_length + self.memory_length * 2)):]
        if binary_cmd == '0111': # jgz
            src, body = self._read_mem_pos(body)
            pos, body = self._read_int(body)
            if trace is not None:
                trace.append('jgz [{}] [{}]'.format(src, pos))
            if self.memory[src] > 0:
                return code[(pos // (self.command_length + self.memory_length * 2) * (self.command_length + self.memory_length * 2)):]
            else:
                body
        if binary_cmd == '1000': # max
            src, body = self._read_mem_pos(body)
            dst, body = self._read_mem_pos(body)
            self.memory[dst] = max(self.memory[src], self.memory[dst])
            if trace is not None:
                trace.append('max [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1001': # min
            src, body = self._read_mem_pos(body)
            dst, body = self._read_mem_pos(body)
            self.memory[dst] = min(self.memory[src], self.memory[dst])
            if trace is not None:
                trace.append('min [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1010': # inc
            dst, body = self._read_mem_pos(body)
            body = self._move_pos(body)
            self.memory[dst] = self.memory[dst] + 1
            if trace is not None:
                trace.append('inc [{}]'.format(dst))
            return body
        if binary_cmd == '1011': # dec
            dst, body = self._read_mem_pos(body)
            body = self._move_pos(body)
            self.memory[dst] = self.memory[dst] - 1
            if trace is not None:
                trace.append('dec [{}]'.format(dst))
            return body
        if binary_cmd == '1100': # mov
            src, body = self._read_mem_pos(body)
            dst, body = self._read_mem_pos(body)
            self.memory[dst] = self.memory[src]
            if trace is not None:
                trace.append('mov [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1101': # bin-max
            src, body = self._read_mem_pos(body)
            dst, body = self._read_mem_pos(body)
            self.memory[dst] = 1 if self.memory[src] > self.memory[dst] else 0
            if trace is not None:
                trace.append('bin-max [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1110': # bin-min
            src, body = self._read_mem_pos(body)
            dst, body = self._read_mem_pos(body)
            self.memory[dst] = 1 if self.memory[src] < self.memory[dst] else 0
            if trace is not None:
                trace.append('bin-min [{}] [{}]'.format(src, dst))
            return body
        if binary_cmd == '1111': # halt
            if trace is not None:
                trace.append('halt')
            return ''
        else: # nope
            body = self._move_pos(body)
            body = self._move_pos(body)
            return body

