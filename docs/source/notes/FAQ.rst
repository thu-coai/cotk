Frequently Asked Questions
==============================


Why my codes seem running from beginning after calling metric.close ?
----------------------------------------------------------------------------------

**TL;DR: Use if __name__ == "__main__" at your start point of program.**

This happened when you use SelfBleuMetric, FwBwBleuMetric on windows. They use
``multiprocess`` for speed issue but this would call ``fork()`` of system. If you are
on windows, which doesn't support ``fork()``, ``Multiprocess`` has to rerun your program
for correctly importing packages.

