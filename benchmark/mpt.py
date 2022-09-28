# coding: utf-8
# 2022/4/25 @ tongshiwei

from tqdm import tqdm
from longling import AsyncLoopIter as _Iter
from longling import concurrent_pool


def produce(data, produce_queue):
    _stop = False
    data = data() if callable(data) else data
    try:
        for _data in data:
            produce_queue.put(_data)
        raise StopIteration

    except StopIteration as e:
        if not _stop:
            _stop = True
            produce_queue.put(e)

    except Exception as e:  # pragma: no cover
        if not _stop:
            _stop = True
            produce_queue.put(e)


class AsyncLoopIter(_Iter):
    def reset(self):
        if self.mode == "p" and callable(self._reset):
            self._set_length()
            self._data = self._reset
            if self.thread is not None:
                self.thread.join()

            self.thread = self.thread_cls(
                target=produce,
                kwargs=dict(data=self._data, produce_queue=self.queue),
                daemon=True
            )
            self.thread.start()
        else:
            super(AsyncLoopIter, self).reset()


def gen():
    return list([i for i in range(5)])


def gather(a):
    return sum(a())


if __name__ == '__main__':
    loader = AsyncLoopIter(gen, level="p")
    for e in range(2):
        for _ in tqdm(loader, "%s" % e):
            pass
