import concurrent.futures
import multiprocessing
import time
import warnings
from tqdm import tqdm

AVAILABLE_CORES = multiprocessing.cpu_count()

print('Cores available:', AVAILABLE_CORES)


class TaskRunner:
    def __init__(self, task, arg_list, max_workers=AVAILABLE_CORES // 2, use_tqdm=True):
        self.max_workers = max_workers
        self.task = task
        self.arg_list = arg_list
        self.use_tqdm = use_tqdm

    def run(self):
        self.now = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._task, arg)
                       for arg in self.arg_list}
            completed = concurrent.futures.as_completed(futures)
            if self.use_tqdm:
                completed = tqdm(completed, total=len(self.arg_list))
            self.results_ = [future.result() for future in completed]
        print("Finished all @%ss" % (time.time() - self.now))

    def _task(self, arg):
        try:
            ret = self.task(arg)
        except Exception as e:
            warnings.warn("TASK ERROR:=====" + str(e) + "=====" + str(arg))
            return "error", arg, None
        if not self.use_tqdm:
            print("Finished %s @%ss" % (arg, time.time() - self.now))
        return "success", arg, ret

    @property
    def errors_(self):
        if not hasattr(self, "results_"):
            raise AttributeError
        return [r[1] for r in self.results_ if r[0] == "error"]


if __name__ == '__main__':
    print("The following provides the usage code of the multi-core `TaskRunner`.")


    def task(x):
        print(x)
        time.sleep(x / 5)
        return x + 1


    now = time.time()
    for i in range(10):
        task(i)
    print("Without multi-processing:", time.time() - now)

    now = time.time()
    runner = TaskRunner(task=task,
                        arg_list=range(10),
                        max_workers=5)
    runner.run()
    print("Results:", runner.results_)
    print("Errors:", runner.errors_)
    print("With multi-processing:", time.time() - now)
