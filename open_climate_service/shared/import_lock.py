"""One lock for loading xclim and the process implementations in more than one thread (CLIM-1549).

Two threads importing xclim for the first time at once can deadlock on Python's per-module
import locks: its indicator modules import each other, so the interpreter raises
``_DeadlockError`` in one thread and leaves a partially initialised module in the other. Startup
loads the process registry in a worker thread, so a request reaching xclim in those seconds (a
template's units checked while listing data sources, say) took part in exactly that race, and
both failed.

Every first import of xclim, and every load of the process implementations that import it,
holds this lock, so a second thread waits for the first to finish rather than racing it. Once the
modules are loaded an import is a dictionary lookup, so the lock costs nothing afterwards. It is
re-entrant because building the registry loads the process plugins too.
"""

import threading

IMPORT_LOCK = threading.RLock()
