# Copyright (C) 2021  Ernie Chang & ...
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import logging
import time

import torch
import torch.nn as nn

from transformers import AdamW



logging.basicConfig(level=logging.INFO)
main_logger = logging.getLogger(__name__)

# load data
import sys
import random

source = list(open(sys.argv[1]))
target_lang = sys.argv[2] #e.g. 'fr'
lang_code = f'{target_lang}_XX'


en_text = "show me round trip fares from denver to philadelphia"
fr_text = "Me montrer les tarifs aller-retour de Denver à Philadelphie ."


def model_builder(rank):

    # define teacher (tc) model
    from transformers import MarianMTModel, MarianTokenizer
    model_name = f"Helsinki-NLP/opus-mt-en-{target_lang}"
    model = MarianMTModel.from_pretrained(model_name).to(rank)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    # build optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    modules = {
        "model": model,
        "tokenizer": tokenizer,
        "optimizer": optimizer
    }

    return modules

def clean(s):
    return s.strip()

def generate(rank,model,tokenizer):

    output_translations = ""
    for sample in source:
        sample = clean(sample) # can be extended
        translated = model.generate(**tokenizer(sample, return_tensors="pt", padding=True).to(rank))
        translation = [tokenizer.decode(t, skip_special_tokens=True) for t in translated][0]
        main_logger.info(f"Source: {sample}, Prediction: {translation}")
        output_translations += f"{sample}\t{translation}\n"

    # write to output file
    with open(f'en_{target_lang}.txt','w') as f:
        f.write(f'{output_translations}')


def batch_iter(src,tgt,batch_size):
    start_index = 0
    while start_index<len(source)-1:
        if start_index+batch_size>len(source)-1:
            yield (src[start_index:-1],tgt[start_index:-1])
        else:
            yield (src[start_index:start_index+batch_size],tgt[start_index:start_index+batch_size])
        start_index += batch_size


def main_translate():

    batch_size = 64
    rank = 'cuda:2'

    modules = model_builder(rank)

    # translate
    generate(rank,modules['model'],modules['tokenizer'])


if __name__ == "__main__":
    main_translate()
