import pytz
from time import mktime
from datetime import datetime
import time
import psycopg2
from psycopg2 import extras as ex
import os.path as osp
import os
import copy
import torch
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import TemporalData
from torch_geometric.nn import TGNMemory, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import LastNeighborLoader, IdentityMessage
# from torch_geometric import *
from tqdm import tqdm
import networkx as nx
import numpy as np
import math
import xxhash
import gc
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from .config import *
except ImportError:  # pragma: no cover
    from config import *


def ns_time_to_datetime(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    seconds, nanos = divmod(int(ns), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=pytz.UTC)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(nanos).zfill(9)
    return s


def ns_time_to_datetime_US(ns):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00.000000000
    """
    tz = pytz.timezone('US/Eastern')
    seconds, nanos = divmod(int(ns), 1_000_000_000)
    dt = datetime.fromtimestamp(seconds, tz=pytz.UTC).astimezone(tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    s += '.' + str(nanos).zfill(9)
    return s


def time_to_datetime_US(s):
    """
    :param ns: int nano timestamp
    :return: datetime   format: 2013-10-10 23:40:00
    """
    tz = pytz.timezone('US/Eastern')
    dt = datetime.fromtimestamp(int(s), tz=pytz.UTC).astimezone(tz)
    s = dt.strftime('%Y-%m-%d %H:%M:%S')

    return s


def datetime_to_ns_time(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = pytz.UTC.localize(dt)
    return int(dt.timestamp() * 1_000_000_000)


def datetime_to_ns_time_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = tz.localize(dt)
    return int(dt.timestamp() * 1_000_000_000)


def datetime_to_timestamp_US(date):
    """
    :param date: str   format: %Y-%m-%d %H:%M:%S   e.g. 2013-10-10 23:40:00
    :return: nano timestamp
    """
    tz = pytz.timezone('US/Eastern')
    dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    dt = tz.localize(dt)
    return int(dt.timestamp())


def init_database_connection():
    if HOST is not None:
        connect = psycopg2.connect(database=DATABASE,
                                   host=HOST,
                                   user=USER,
                                   password=PASSWORD,
                                   port=PORT
                                   )
    else:
        connect = psycopg2.connect(database=DATABASE,
                                   user=USER,
                                   password=PASSWORD,
                                   port=PORT
                                   )
    cur = connect.cursor()
    return cur, connect


def gen_nodeid2msg(cur):
    sql = "select * from node2id ORDER BY index_id;"
    cur.execute(sql)
    rows = cur.fetchall()
    nodeid2msg = {}
    for i in rows:
        nodeid2msg[i[0]] = i[-1]
        nodeid2msg[i[-1]] = {i[1]: i[2]}

    return nodeid2msg


def tensor_find(t, x):
    t_np = t.cpu().numpy()
    idx = np.argwhere(t_np == x)
    return idx[0][0]+1


def std(t):
    t = np.array(t)
    return np.std(t)


def var(t):
    t = np.array(t)
    return np.var(t)


def mean(t):
    t = np.array(t)
    return np.mean(t)


def _timestamp_str_to_ns(timestamp: str) -> Optional[int]:
    if not timestamp:
        return None
    if "." in timestamp:
        base, frac = timestamp.split(".", 1)
    else:
        base, frac = timestamp, ""
    try:
        base_ns = datetime_to_ns_time_US(base)
    except ValueError:
        return None
    frac_digits = "".join(ch for ch in frac if ch.isdigit())
    if not frac_digits:
        return base_ns
    frac_digits = (frac_digits + "000000000")[:9]
    return base_ns + int(frac_digits)


def _filename_interval_to_ns(path: Path) -> Optional[Tuple[int, int]]:
    stem = path.stem
    if "~" not in stem:
        return None
    start_raw, end_raw = stem.split("~", 1)
    start_ns = _timestamp_str_to_ns(start_raw)
    end_ns = _timestamp_str_to_ns(end_raw)
    if start_ns is None or end_ns is None:
        return None
    return start_ns, end_ns


"""
    List of attack files to be investigated
    The time windows of the following attacks are as follows:
    2018_04_06 11_00 - 2018_04_06 12_08
    2018_04_11 15_08 - 2018_04_11 15_15
    2018_04_12 14_00 - 2018_04_12 14_38
    2018_04_13 09_04 - 2018_04_13 09_15
"""


def fetch_attack_list() -> List[str]:
    """
    Collect log windows that overlap the 2018-04-06 11:00-12:08 EDT attack window.
    Comparisons are done in epoch nanoseconds.
    """
    base_dir = Path(ARTIFACT_DIR)
    if not base_dir.exists():
        return []

    window_start = datetime_to_ns_time_US("2018-04-06 11:00:00")
    window_end = datetime_to_ns_time_US("2018-04-06 12:08:00")

    matches: List[str] = []
    for graph_dir in sorted(base_dir.iterdir()):
        if not graph_dir.is_dir() or not graph_dir.name.startswith("graph_"):
            continue
        for candidate in sorted(graph_dir.glob("*.txt")):
            interval = _filename_interval_to_ns(candidate)
            if interval is None:
                continue
            start_ns, end_ns = interval
            if start_ns <= window_end and end_ns >= window_start:
                matches.append(str(candidate))
    # print(f"Found {len(matches)} attack log windows.")
    # print("\n".join(matches))
    return matches


def hashgen(l):
    """Generate a single hash value from a list. @l is a list of
    string values, which can be properties of a node/edge. This
    function returns a single hashed integer value."""
    hasher = xxhash.xxh64()
    for e in l:
        hasher.update(e)
    return hasher.intdigest()
