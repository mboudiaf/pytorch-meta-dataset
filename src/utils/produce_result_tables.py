import pandas as pd
from markdownTable import markdownTable
from loguru import logger
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from functools import partial
from typing import Dict, List, Any
# file = 'sample.csv'
# df = pd.read_csv(file)
# markdownTable(data).setParams(row_sep = 'markdown', quote = False).getMarkdown()


def main() -> None:

    # Recover all files that match .npy pattern in folder/
    #  ===== Recover all csv result files =====
    p = Path('results')
    csv_files = list(p.glob('**/*.csv'))  # every csv file represents the  summary of an experiment
    csv_files.sort()
    assert len(csv_files)

    #  ===== Recover all csv result files =====
    result_dict = nested_default_dict(2, list)

    for file in csv_files:
        df = pd.read_csv(file)

        # Read remaining rows and add it to result_dict
        for index, row in df.iterrows():
            result_dict[row['method']][row['test_source']] = np.round(100 * row['acc'], 2)

    records: List[Dict[str, Any]] = []
    for method in result_dict:
        rec = {'Method': method}
        for dataset, acc in result_dict[method].items():
            rec[dataset] = acc
        records.append(rec)

    mt = markdownTable(records).setParams(row_sep='markdown', quote=False).getMarkdown()
    logger.info(mt)


def nested_default_dict(depth: int, final_type: Any, i: int = 1):
    if i == depth:
        return defaultdict(final_type)
    fn = partial(nested_default_dict, depth=depth, final_type=final_type, i=i+1)
    return defaultdict(fn)


if __name__ == "__main__":
    main()
