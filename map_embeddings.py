# Copyright (C) 2016-2018  Mikel Artetxe <artetxem@gmail.com>
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


import embeddings
import eval_translation
from cupy_utils import *

import argparse
import collections

import re
import sys
import time
import os
from datetime import datetime

import scipy
import numpy as np
import pandas as pd

import torch
import pymanopt
#import pymanopt.autograd.numpy as np
from pymanopt.manifolds import Stiefel, SymmetricPositiveDefinite, Product
from pymanopt.optimizers import ConjugateGradient


def dropout(m, p):
    if p <= 0.0:
        return m
    else:
        xp = get_array_module(m)
        mask = xp.random.rand(*m.shape) >= p
        return m*mask


def topk_mean(m, k, inplace=False):  # TODO Assuming that axis is 1
    xp = get_array_module(m)
    n = m.shape[0]
    ans = xp.zeros(n, dtype=m.dtype)
    if k <= 0:
        return ans
    if not inplace:
        m = xp.array(m)
    ind0 = xp.arange(n)
    ind1 = xp.empty(n, dtype=int)
    minimum = m.min()
    for i in range(k):
        m.argmax(axis=1, out=ind1)
        ans += m[ind0, ind1]
        m[ind0, ind1] = minimum
    return ans / k

randomGenerator = np.random.default_rng()

def Orthogonal(D1, D2):
    #G = np.random.randn(D, D).astype('float32')
    #G = np.random.Generator.standard_normal(size=(D, D)).astype(np.float32)
    #G = np.random.Generator.standard_normal(size=(D, D), dtype=np.float64, out=None)

    G = randomGenerator.standard_normal(size=(D1, D2), dtype=np.float32, out=None)
    Q, _ = np.linalg.qr(G)

    return Q

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Map word embeddings in two languages into a shared space')
    parser.add_argument('src_input', help='the input source embeddings')
    parser.add_argument('trg_input', help='the input target embeddings')
    parser.add_argument('src_output', help='the output source embeddings')
    parser.add_argument('trg_output', help='the output target embeddings')
    parser.add_argument('--encoding', default='utf-8', help='the character encoding for input/output (defaults to utf-8)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32', help='the floating-point precision (defaults to fp32)')
    parser.add_argument('--cuda', action='store_true', help='use cuda (requires cupy)')
    parser.add_argument('--batch_size', default=10000, type=int, help='batch size (defaults to 10000); does not affect results, larger is usually faster but uses more memory')
    parser.add_argument('--seed', type=int, default=0, help='the random seed (defaults to 0)')

    recommended_group = parser.add_argument_group('recommended settings', 'Recommended settings for different scenarios')
    recommended_type = recommended_group.add_mutually_exclusive_group()
    recommended_type.add_argument('--supervised', metavar='DICTIONARY', help='recommended if you have a large training dictionary')
    recommended_type.add_argument('--semi_supervised', metavar='DICTIONARY', help='recommended if you have a small seed dictionary')
    recommended_type.add_argument('--identical', action='store_true', help='recommended if you have no seed dictionary but can rely on identical words')
    recommended_type.add_argument('--unsupervised', action='store_true', help='recommended if you have no seed dictionary and do not want to rely on identical words')
    recommended_type.add_argument('--acl2018', action='store_true', help='reproduce our ACL 2018 system')
    recommended_type.add_argument('--aaai2018', metavar='DICTIONARY', help='reproduce our AAAI 2018 system')
    recommended_type.add_argument('--acl2017', action='store_true', help='reproduce our ACL 2017 system with numeral initialization')
    recommended_type.add_argument('--acl2017_seed', metavar='DICTIONARY', help='reproduce our ACL 2017 system with a seed dictionary')
    recommended_type.add_argument('--emnlp2016', metavar='DICTIONARY', help='reproduce our EMNLP 2016 system')

    init_group = parser.add_argument_group('advanced initialization arguments', 'Advanced initialization arguments')
    init_type = init_group.add_mutually_exclusive_group()
    init_type.add_argument('-d', '--init_dictionary', default=sys.stdin.fileno(), metavar='DICTIONARY', help='the training dictionary file (defaults to stdin)')
    init_type.add_argument('--init_identical', action='store_true', help='use identical words as the seed dictionary')
    init_type.add_argument('--init_numerals', action='store_true', help='use latin numerals (i.e. words matching [0-9]+) as the seed dictionary')
    init_type.add_argument('--init_unsupervised', action='store_true', help='use unsupervised initialization')
    init_group.add_argument('--unsupervised_vocab', type=int, default=0, help='restrict the vocabulary to the top k entries for unsupervised initialization')

    mapping_group = parser.add_argument_group('advanced mapping arguments', 'Advanced embedding mapping arguments')
    mapping_group.add_argument('--normalize', choices=['unit', 'center', 'unitdim', 'centeremb', 'none'], nargs='*', default=[], help='the normalization actions to perform in order')
    mapping_group.add_argument('--whiten', action='store_true', help='whiten the embeddings')
    mapping_group.add_argument('--src_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the source language embeddings')
    mapping_group.add_argument('--trg_reweight', type=float, default=0, nargs='?', const=1, help='re-weight the target language embeddings')
    mapping_group.add_argument('--src_dewhiten', choices=['src', 'trg'], help='de-whiten the source language embeddings')
    mapping_group.add_argument('--trg_dewhiten', choices=['src', 'trg'], help='de-whiten the target language embeddings')
    mapping_group.add_argument('--dim_reduction', type=int, default=0, help='apply dimensionality reduction')

    # GEOMM enhancement args
    geomm_group = parser.add_argument_group('Additional for geomm','Additional for geomm')
    geomm_group.add_argument('-ns', '--new_space', action='store_true', help='if orthogonal constrained, mapping is to a new space')
    geomm_group.add_argument('--geomm', action='store_true', help='run geomm in last iteration')
    geomm_group.add_argument('--l2_reg', type=float, default=1e2, help='Lambda for L2 Regularization')
    geomm_group.add_argument('--max_opt_time', type=int, default=5000, help='Maximum time limit for optimization in seconds')
    geomm_group.add_argument('--max_opt_iter', type=int, default=150, help='Maximum number of iterations for optimization')
    geomm_group.add_argument('--no_whiten', action='store_true', help='force no-whiten the embeddings')
    geomm_group.add_argument('--no_reweight', action='store_true', help='force no-reweight the embeddings')
    geomm_group.add_argument('--full_iter_geomm', action='store_true', help='run geomm in iterations')
    geomm_group.add_argument('--eval_translation', action='store_true', help='run eval translation at the end')
    geomm_group.add_argument('--log_results_file', help='when eval translation log to the file')

    # end of GEOMM enhancement args

    mapping_type = mapping_group.add_mutually_exclusive_group()
    mapping_type.add_argument('-c', '--orthogonal', action='store_true', help='use orthogonal constrained mapping')
    mapping_type.add_argument('-u', '--unconstrained', action='store_true', help='use unconstrained mapping')

    self_learning_group = parser.add_argument_group('advanced self-learning arguments', 'Advanced arguments for self-learning')
    self_learning_group.add_argument('--self_learning', action='store_true', help='enable self-learning')
    self_learning_group.add_argument('--vocabulary_cutoff', type=int, default=0, help='restrict the vocabulary to the top k entries')
    self_learning_group.add_argument('--direction', choices=['forward', 'backward', 'union'], default='union', help='the direction for dictionary induction (defaults to union)')
    self_learning_group.add_argument('--csls', type=int, nargs='?', default=0, const=10, metavar='NEIGHBORHOOD_SIZE', dest='csls_neighborhood', help='use CSLS for dictionary induction')
    self_learning_group.add_argument('--threshold', default=0.000001, type=float, help='the convergence threshold (defaults to 0.000001)')
    self_learning_group.add_argument('--validation', default=None, metavar='DICTIONARY', help='a dictionary file for validation at each iteration')
    self_learning_group.add_argument('--stochastic_initial', default=0.1, type=float, help='initial keep probability stochastic dictionary induction (defaults to 0.1)')
    self_learning_group.add_argument('--stochastic_multiplier', default=2.0, type=float, help='stochastic dictionary induction multiplier (defaults to 2.0)')
    self_learning_group.add_argument('--stochastic_interval', default=50, type=int, help='stochastic dictionary induction interval (defaults to 50)')
    self_learning_group.add_argument('--log', help='write to a log file in tsv format at each iteration')
    self_learning_group.add_argument('-v', '--verbose', action='store_true', help='write log information to stderr at each iteration')
    args = parser.parse_args()

    if args.supervised is not None:
        parser.set_defaults(init_dictionary=args.supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.semi_supervised is not None:
        parser.set_defaults(init_dictionary=args.semi_supervised, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.identical:
        parser.set_defaults(init_identical=True, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.unsupervised or args.acl2018:
        parser.set_defaults(init_unsupervised=True, unsupervised_vocab=4000, normalize=['unit', 'center', 'unit'], whiten=True, src_reweight=0.5, trg_reweight=0.5, src_dewhiten='src', trg_dewhiten='trg', self_learning=True, vocabulary_cutoff=20000, csls_neighborhood=10)
    if args.aaai2018:
        parser.set_defaults(init_dictionary=args.aaai2018, normalize=['unit', 'center'], whiten=True, trg_reweight=1, src_dewhiten='src', trg_dewhiten='trg', batch_size=1000)
    if args.acl2017:
        parser.set_defaults(init_numerals=True, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.acl2017_seed:
        parser.set_defaults(init_dictionary=args.acl2017_seed, orthogonal=True, normalize=['unit', 'center'], self_learning=True, direction='forward', stochastic_initial=1.0, stochastic_interval=1, batch_size=1000)
    if args.emnlp2016:
        parser.set_defaults(init_dictionary=args.emnlp2016, orthogonal=True, normalize=['unit', 'center'], batch_size=1000)
    args = parser.parse_args()

    if args.no_whiten:
        args.whiten = False
        args.src_dewhiten = None
        args.trg_dewhiten = None

    main_start = time.time()
    print(f'Current time : {time.strftime("%H:%M:%S")}')
    print(f'args.whiten : {args.whiten}')
    print(f'args.src_dewhiten : {args.src_dewhiten}')
    print(f'args.trg_dewhiten : {args.trg_dewhiten}')
    print(f'args.direction : {args.direction}')

    # Check command line arguments
    if (args.src_dewhiten is not None or args.trg_dewhiten is not None) and not args.whiten:
        print('ERROR: De-whitening requires whitening first', file=sys.stderr)
        sys.exit(-1)

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'

    # Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)

    # NumPy/CuPy management
    if args.cuda:
        if not supports_cupy():
            print('ERROR: Install CuPy for CUDA support', file=sys.stderr)
            sys.exit(-1)
        xp = get_cupy()
        x = xp.asarray(x)
        z = xp.asarray(z)
    else:
        xp = np
    xp.random.seed(args.seed)
    np.random.seed(args.seed)

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    # STEP 0: Normalization
    embeddings.normalize(x, args.normalize)
    embeddings.normalize(z, args.normalize)

    # Build the seed dictionary
    src_indices = []
    trg_indices = []
    if args.init_unsupervised:
        sim_size = min(x.shape[0], z.shape[0]) if args.unsupervised_vocab <= 0 else min(x.shape[0], z.shape[0], args.unsupervised_vocab)
        u, s, vt = xp.linalg.svd(x[:sim_size], full_matrices=False)
        xsim = (u*s).dot(u.T)
        u, s, vt = xp.linalg.svd(z[:sim_size], full_matrices=False)
        zsim = (u*s).dot(u.T)
        del u, s, vt
        xsim.sort(axis=1)
        zsim.sort(axis=1)
        embeddings.normalize(xsim, args.normalize)
        embeddings.normalize(zsim, args.normalize)
        sim = xsim.dot(zsim.T)
        if args.csls_neighborhood > 0:
            knn_sim_fwd = topk_mean(sim, k=args.csls_neighborhood)
            knn_sim_bwd = topk_mean(sim.T, k=args.csls_neighborhood)
            sim -= knn_sim_fwd[:, xp.newaxis]/2 + knn_sim_bwd/2
        if args.direction == 'forward':
            src_indices = xp.arange(sim_size)
            trg_indices = sim.argmax(axis=1)
        elif args.direction == 'backward':
            src_indices = sim.argmax(axis=0)
            trg_indices = xp.arange(sim_size)
        elif args.direction == 'union':
            src_indices = xp.concatenate((xp.arange(sim_size), sim.argmax(axis=0)))
            trg_indices = xp.concatenate((sim.argmax(axis=1), xp.arange(sim_size)))
        del xsim, zsim, sim
    elif args.init_numerals:
        numeral_regex = re.compile('^[0-9]+$')
        src_numerals = {word for word in src_words if numeral_regex.match(word) is not None}
        trg_numerals = {word for word in trg_words if numeral_regex.match(word) is not None}
        numerals = src_numerals.intersection(trg_numerals)
        for word in numerals:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    elif args.init_identical:
        identical = set(src_words).intersection(set(trg_words))
        for word in identical:
            src_indices.append(src_word2ind[word])
            trg_indices.append(trg_word2ind[word])
    else:
        f = open(args.init_dictionary, encoding=args.encoding, errors='surrogateescape')
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                src_indices.append(src_ind)
                trg_indices.append(trg_ind)
            except KeyError:
                print('WARNING: OOV dictionary entry ({0} - {1})'.format(src, trg), file=sys.stderr)

    # Read validation dictionary
    if args.validation is not None:
        f = open(args.validation, encoding=args.encoding, errors='surrogateescape')
        validation = collections.defaultdict(set)
        oov = set()
        vocab = set()
        for line in f:
            src, trg = line.split()
            try:
                src_ind = src_word2ind[src]
                trg_ind = trg_word2ind[trg]
                validation[src_ind].add(trg_ind)
                vocab.add(src)
            except KeyError:
                oov.add(src)
        oov -= vocab  # If one of the translation options is in the vocabulary, then the entry is not an oov
        validation_coverage = len(validation) / (len(validation) + len(oov))

    # Create log file
    if args.log:
        log = open(args.log, mode='w', encoding=args.encoding, errors='surrogateescape')

    # Allocate memory
    xw = xp.empty_like(x)
    zw = xp.empty_like(z)
    src_size = x.shape[0] if args.vocabulary_cutoff <= 0 else min(x.shape[0], args.vocabulary_cutoff)
    trg_size = z.shape[0] if args.vocabulary_cutoff <= 0 else min(z.shape[0], args.vocabulary_cutoff)
    simfwd = xp.empty((args.batch_size, trg_size), dtype=dtype)
    simbwd = xp.empty((args.batch_size, src_size), dtype=dtype)
    if args.validation is not None:
        simval = xp.empty((len(validation.keys()), z.shape[0]), dtype=dtype)

    best_sim_forward = xp.full(src_size, -100, dtype=dtype)
    src_indices_forward = xp.arange(src_size)
    trg_indices_forward = xp.zeros(src_size, dtype=int)
    best_sim_backward = xp.full(trg_size, -100, dtype=dtype)
    src_indices_backward = xp.zeros(trg_size, dtype=int)
    trg_indices_backward = xp.arange(trg_size)
    knn_sim_fwd = xp.zeros(src_size, dtype=dtype)
    knn_sim_bwd = xp.zeros(trg_size, dtype=dtype)

    # Training loop
    best_objective = objective = -100.
    it = 1
    last_improvement = 0
    keep_prob = args.stochastic_initial
    t = time.time()
    end = not args.self_learning

    xw[:] = x
    zw[:] = z

    while True:

        # Increase the keep probability if we have not improve in args.stochastic_interval iterations
        if it - last_improvement > args.stochastic_interval:
            if keep_prob >= 1.0:
                end = True
            keep_prob = min(1.0, args.stochastic_multiplier*keep_prob)
            last_improvement = it

        # Update the embedding mapping
        if args.orthogonal or not end:  # orthogonal mapping

            if args.full_iter_geomm:
                # GEOMM
                #print(f"Running GEOMM in all iterations")
                xw, zw, _, _ = opt_geomm_fast(x, z, xw, zw, src_indices, trg_indices, xp, args, dtype)
            else:
                # Vecmap
                #print(f"Running SVD in all iterations")
                u, s, vt = xp.linalg.svd(z[trg_indices].T.dot(x[src_indices]))
                w = vt.T.dot(u.T)
                x.dot(w, out=xw)
                zw[:] = z

        elif args.unconstrained:  # unconstrained mapping
            print(f"Running unconstrained:{args.unconstrained}")
            x_pseudoinv = xp.linalg.inv(x[src_indices].T.dot(x[src_indices])).dot(x[src_indices].T)
            w = x_pseudoinv.dot(z[trg_indices])
            x.dot(w, out=xw)
            zw[:] = z
        else:  # advanced mapping

            print("\nRunning final transformation!!!")
            # TODO xw.dot(wx2, out=xw) and alike not working
            xw[:] = x
            zw[:] = z

            # STEP 1: Whitening
            def whitening_transformation(m):
                u, s, vt = xp.linalg.svd(m, full_matrices=False)
                return vt.T.dot(xp.diag(1/s)).dot(vt)
            if args.whiten:
                print(f"Running whiten:{args.whiten}")
                wx1 = whitening_transformation(xw[src_indices])
                wz1 = whitening_transformation(zw[trg_indices])
                xw = xw.dot(wx1)
                zw = zw.dot(wz1)

            # STEP 2: Orthogonal mapping
            if args.geomm:
                # GEOMM
                print(f"Running GEOMM. GEOMM:{args.geomm}")
                xw, zw, wx2, wz2 = opt_geomm_fast(x, z, xw, zw, src_indices, trg_indices, xp, args, dtype)

                if args.whiten or (not args.no_reweight):
                    # Calc s and wx2, wz2_t.
                    print(f"Calculating s and wx2, wz2_t for whiten or reweight")
                    wx2, s, wz2_t = xp.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
                    wz2 = wz2_t.T
            else:
                # Vecmap
                print(f"Running Vecmap. GEOMM:{args.geomm}")
                wx2, s, wz2_t = xp.linalg.svd(xw[src_indices].T.dot(zw[trg_indices]))
                wz2 = wz2_t.T
                xw = xw.dot(wx2)
                zw = zw.dot(wz2)


            # STEP 3: Re-weighting
            if not args.no_reweight:
                print(f"Running reweight")
                xw *= s**args.src_reweight
                zw *= s**args.trg_reweight

            # STEP 4: De-whitening
            if args.src_dewhiten == 'src':
                print(f"Running src_dewhiten:{args.src_dewhiten}")
                xw = xw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.src_dewhiten == 'trg':
                print(f"Running src_dewhiten:{args.src_dewhiten}")
                xw = xw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))
            if args.trg_dewhiten == 'src':
                print(f"Running trg_dewhiten:{args.trg_dewhiten}")
                zw = zw.dot(wx2.T.dot(xp.linalg.inv(wx1)).dot(wx2))
            elif args.trg_dewhiten == 'trg':
                print(f"Running trg_dewhiten:{args.trg_dewhiten}")
                zw = zw.dot(wz2.T.dot(xp.linalg.inv(wz1)).dot(wz2))

            # STEP 5: Dimensionality reduction
            if args.dim_reduction > 0:
                print(f"Running dim_reduction:{args.dim_reduction}")
                xw = xw[:, :args.dim_reduction]
                zw = zw[:, :args.dim_reduction]

        # Self-learning
        if end:
            break
        else:
            # Update the training dictionary
            if args.direction in ('forward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, trg_size, simbwd.shape[0]):
                        j = min(i + simbwd.shape[0], trg_size)
                        zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                        knn_sim_bwd[i:j] = topk_mean(simbwd[:j-i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, src_size, simfwd.shape[0]):
                    j = min(i + simfwd.shape[0], src_size)
                    xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                    simfwd[:j-i].max(axis=1, out=best_sim_forward[i:j])
                    simfwd[:j-i] -= knn_sim_bwd/2  # Equivalent to the real CSLS scores for NN
                    dropout(simfwd[:j-i], 1 - keep_prob).argmax(axis=1, out=trg_indices_forward[i:j])
            if args.direction in ('backward', 'union'):
                if args.csls_neighborhood > 0:
                    for i in range(0, src_size, simfwd.shape[0]):
                        j = min(i + simfwd.shape[0], src_size)
                        xw[i:j].dot(zw[:trg_size].T, out=simfwd[:j-i])
                        knn_sim_fwd[i:j] = topk_mean(simfwd[:j-i], k=args.csls_neighborhood, inplace=True)
                for i in range(0, trg_size, simbwd.shape[0]):
                    j = min(i + simbwd.shape[0], trg_size)
                    zw[i:j].dot(xw[:src_size].T, out=simbwd[:j-i])
                    simbwd[:j-i].max(axis=1, out=best_sim_backward[i:j])
                    simbwd[:j-i] -= knn_sim_fwd/2  # Equivalent to the real CSLS scores for NN
                    dropout(simbwd[:j-i], 1 - keep_prob).argmax(axis=1, out=src_indices_backward[i:j])
            if args.direction == 'forward':
                src_indices = src_indices_forward
                trg_indices = trg_indices_forward
            elif args.direction == 'backward':
                src_indices = src_indices_backward
                trg_indices = trg_indices_backward
            elif args.direction == 'union':
                src_indices = xp.concatenate((src_indices_forward, src_indices_backward))
                trg_indices = xp.concatenate((trg_indices_forward, trg_indices_backward))

            # Objective function evaluation
            if args.direction == 'forward':
                objective = xp.mean(best_sim_forward).tolist()
            elif args.direction == 'backward':
                objective = xp.mean(best_sim_backward).tolist()
            elif args.direction == 'union':
                objective = (xp.mean(best_sim_forward) + xp.mean(best_sim_backward)).tolist() / 2
            if objective - best_objective >= args.threshold:
                last_improvement = it
                best_objective = objective

            # Accuracy and similarity evaluation in validation
            if args.validation is not None:
                src = list(validation.keys())
                xw[src].dot(zw.T, out=simval)
                nn = asnumpy(simval.argmax(axis=1))
                accuracy = np.mean([1 if nn[i] in validation[src[i]] else 0 for i in range(len(src))])
                similarity = np.mean([max([simval[i, j].tolist() for j in validation[src[i]]]) for i in range(len(src))])

            # Logging
            duration = time.time() - t
            if args.verbose:
                print(file=sys.stderr)
                print('ITERATION {0} ({1:.2f}s)'.format(it, duration), file=sys.stderr)
                print('\t- Objective:        {0:9.4f}%'.format(100 * objective), file=sys.stderr)
                print('\t- Drop probability: {0:9.4f}%'.format(100 - 100*keep_prob), file=sys.stderr)
                if args.validation is not None:
                    print('\t- Val. similarity:  {0:9.4f}%'.format(100 * similarity), file=sys.stderr)
                    print('\t- Val. accuracy:    {0:9.4f}%'.format(100 * accuracy), file=sys.stderr)
                    print('\t- Val. coverage:    {0:9.4f}%'.format(100 * validation_coverage), file=sys.stderr)
                sys.stderr.flush()
            if args.log is not None:
                val = '{0:.6f}\t{1:.6f}\t{2:.6f}'.format(
                    100 * similarity, 100 * accuracy, 100 * validation_coverage) if args.validation is not None else ''
                print('{0}\t{1:.6f}\t{2}\t{3:.6f}'.format(it, 100 * objective, val, duration), file=log)
                log.flush()

        t = time.time()
        it += 1

    # Write mapped embeddings
    print(f'\nWriting embeddings...')
    srcfile = open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(src_words, xw, srcfile)
    embeddings.write(trg_words, zw, trgfile)
    srcfile.close()
    trgfile.close()

    main_end = time.time()

    if args.eval_translation:
        print(f'Running eval translation...')
        translation_start = time.time()

        retrieval_value = 'csls'

        # Define the arguments as a list of strings
        eval_args = [
            args.src_output,
            args.trg_output,
            '-d', args.validation,
            '--retrieval', retrieval_value,
            '--cuda',
        ]

        print("\nCalling the evaluation script...")
        eval_result = eval_translation.main(eval_args)
        print(f"Evaluation script returned: {eval_result}")
        translation_end = time.time()

        if args.log_results_file:

            if args.acl2018:
                recommended_type_value = 'acl2018'
            elif args.unsupervised:
                recommended_type_value = 'unsupervised'
            else:
                recommended_type_value = None

            run_settings = {
                'source': os.path.basename(args.src_input),
                'target': os.path.basename(args.trg_input),
                'dictionary': os.path.basename(args.validation),
                'type': recommended_type_value,
                'seed': args.seed,
                'geomm': args.geomm,
                'no_whiten': args.no_whiten,
                'no_reweight': args.no_reweight,
                'full_iter_geomm': args.full_iter_geomm,
                'l2_reg': args.l2_reg,
                'max_opt_time': args.max_opt_time,
                'max_opt_iter': args.max_opt_iter,
                'eval_retrieval': retrieval_value
            }

            eval_result['iterations'] = it
            eval_result['translation_seconds'] = translation_end -translation_start
            eval_result['transformation_seconds'] = main_end - main_start
            eval_result['coverage_%'] = f"{eval_result['coverage']:7.2%}"
            eval_result['accuracy_%'] = f"{eval_result['accuracy']:7.2%}"
            log_evaluation_results(log_file=args.log_results_file, settings=run_settings, results=eval_result)


def log_evaluation_results(log_file, settings, results):
    """
    Logs experiment settings and results to a CSV file.

    If a row with the same settings already exists, it's overwritten
    with new results and a new timestamp. Otherwise, a new row is appended.

    Args:
        log_file (str): Path to the CSV log file.
        settings (dict): Dictionary of settings that identify the run (e.g., l2_reg).
        results (dict): Dictionary of results to log (e.g., accuracy).
    """
    # 1. Prepare the new row data
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_row_data = {**settings, **results, 'timestamp': timestamp}

    # 2. Read the existing log file or create a new DataFrame
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        # Create an empty DataFrame with the correct columns if file doesn't exist
        df = pd.DataFrame(columns=list(new_row_data.keys()))

    # 3. Find if a row with these exact settings already exists
    mask = pd.Series([True] * len(df))
    for key, value in settings.items():
        # Handle cases where the column might not exist yet
        if key in df.columns:
            mask &= (df[key] == value)
        else:
            mask = pd.Series([False] * len(df))  # No match if column is missing
            break

    matching_indices = df[mask].index

    # 4. Overwrite or Append
    if not matching_indices.empty:
        # A. Match found: Overwrite the first matching row
        index_to_update = matching_indices[0]
        for key, value in new_row_data.items():
            df.loc[index_to_update, key] = value
        print(f"Log file updated for settings: {settings}")
    else:
        # A. Create a DataFrame for the new row
        new_row_df = pd.DataFrame([new_row_data])

        # B. Check if the original DataFrame is empty
        if df.empty:
            # If so, the new DataFrame is just the new row
            df = new_row_df
        else:
            # Otherwise, concatenate as before
            df = pd.concat([df, new_row_df], ignore_index=True)
        # --- END OF FIX ---
        print(f"New entry added to log for settings: {settings}")

    # 5. Save the updated DataFrame back to CSV
    df.to_csv(log_file, index=False)

def get_geomm_like_A(src_indices, trg_indices):

    x_count = len(set(src_indices))
    z_count = len(set(trg_indices))
    A = np.zeros((x_count, z_count))

    # Creating dictionary matrix from training set
    map_dict_src = {}
    map_dict_trg = {}
    I = 0
    uniq_src = []
    uniq_trg = []
    for i in range(len(src_indices)):
        if src_indices[i] not in map_dict_src.keys():
            map_dict_src[src_indices[i]] = I
            I += 1
            uniq_src.append(src_indices[i])
    J = 0
    for j in range(len(trg_indices)):
        if trg_indices[j] not in map_dict_trg.keys():
            map_dict_trg[trg_indices[j]] = J
            J += 1
            uniq_trg.append(trg_indices[j])

    for i in range(len(src_indices)):
        A[map_dict_src[src_indices[i]], map_dict_trg[trg_indices[i]]] = 1

    return A, uniq_src, uniq_trg

def get_full_dic_geomm_like_A(src_indices, trg_indices):

    x_count = len(src_indices)
    z_count = len(trg_indices)
    A = np.zeros((x_count, z_count))

    # Creating dictionary matrix from training set
    for i in range(len(src_indices)):
        A[src_indices[i], trg_indices[i]] = 1

    return A, src_indices, trg_indices


def opt_geomm(x, z, xw, zw, src_indices, trg_indices, xp, args, dtype):

    print(f'Source indices length : {len(src_indices)},{len(trg_indices)}')
    start = time.time()

    # Using original method of GEOMM.
    #np_A, uniq_src, uniq_trg = get_geomm_like_A(src_indices.get(), trg_indices.get())
    np_A, uniq_src, uniq_trg = get_full_dic_geomm_like_A(src_indices.get(), trg_indices.get())
    xp_x_src = x[uniq_src]
    xp_z_trg = z[uniq_trg]
    print(f'Shapes, np_A:{np_A.shape}, xp_x_src:{xp_x_src.shape}, xp_z_trg:{xp_z_trg.shape}')

    Lambda = args.l2_reg

    # 1. Define the manifold (this part remains very similar)
    print(f'Before manifold definition')
    manifold = Product([
        Stiefel(x.shape[1], x.shape[1]),
        Stiefel(z.shape[1], x.shape[1]),
        SymmetricPositiveDefinite(x.shape[1])  # used to be PositiveDefinite
    ])

    # 2. Define the cost as a standard Python function
    # The decorator tells pymanopt to use autograd to compute gradients automatically
    # @pymanopt.function.autograd(manifold)
    @pymanopt.function.pytorch(manifold)
    def cost(U1, U2, B):
        """
        Defines the cost function using standard NumPy-like operations.
        The variables U1, U2, and B will be passed by the solver.
        """

        # Move to pure torch.
        cost_x_src = torch.from_numpy(xp_x_src.get()).to(B.device, dtype=torch.float64)
        cost_z_trg = torch.from_numpy(xp_z_trg.get()).to(B.device, dtype=torch.float64)
        cost_A = torch.from_numpy(np_A).to(B.device, dtype=torch.float64)

        # Reconstruct the affinity matrix using the variables.
        # Note: Using '@' for matrix multiplication is cleaner
        reconstructed_A = (cost_x_src @ U1 @ B @ U2.T) @ cost_z_trg.T

        # Calculate the two components of the cost.
        reconstruction_error = torch.sum((reconstructed_A - cost_A) ** 2)
        regularization_term = 0.5 * Lambda * torch.sum(B ** 2)

        return reconstruction_error + regularization_term

    # 3. Instantiate the pymanopt problem
    # We pass the manifold and the cost function directly
    print(f'Before pymanopt.Problem')
    problem = pymanopt.Problem(manifold=manifold, cost=cost)

    # 4. Define and run the solver (this part is unchanged)
    print(f'Before ConjugateGradient')
    solver = ConjugateGradient(max_time=15000, max_iterations=150)  # max_iterations=args.max_opt_iter)
    wopt = solver.run(problem)

    # 5. Unpack the optimized weights
    U1, U2, B = wopt.point

    # Step 2: Transformation
    sqrtm_B = xp.asarray(scipy.linalg.sqrtm(B), dtype=dtype)
    U1 = xp.asarray(U1, dtype=dtype)
    U2 = xp.asarray(U2, dtype=dtype)

    result_xw2 = U1.dot(sqrtm_B)
    result_zw2 = U2.dot(sqrtm_B)

    result_xw = x.dot(U1).dot(sqrtm_B)
    result_zw = z.dot(U2).dot(sqrtm_B)

    end = time.time()
    print(f"Geomm runtime: {end - start} seconds")

    return result_xw, result_zw, result_xw2, result_zw2

def opt_geomm_fast(x, z, xw, zw, src_indices, trg_indices, xp, args, dtype):

    print(f'Source indices length : {len(src_indices)},{len(trg_indices)}')
    start = time.time()

    # Using original method of GEOMM.
    np_A, uniq_src, uniq_trg = get_geomm_like_A(src_indices.get(), trg_indices.get())
    xp_x_src = x[uniq_src]
    xp_z_trg = z[uniq_trg]
    print(f'Shapes, np_A:{np_A.shape}, xp_x_src:{xp_x_src.shape}, xp_z_trg:{xp_z_trg.shape}')

    Lambda = args.l2_reg

    print("Pre-computing matrices XtX, ZtZ, and XtAZ...")
    x_src_np = xp_x_src.get()  # Get NumPy array from CuPy
    z_trg_np = xp_z_trg.get()  # Get NumPy array from CuPy

    XtX = x_src_np.T @ x_src_np
    ZtZ = z_trg_np.T @ z_trg_np
    XtAZ = x_src_np.T @ np_A @ z_trg_np

    print("Pre-computation finished.")

    # 1. Define the manifold (this part remains very similar)
    print(f'Before manifold definition')
    manifold = Product([
        Stiefel(x.shape[1], x.shape[1]),
        Stiefel(z.shape[1], x.shape[1]),
        SymmetricPositiveDefinite(x.shape[1])  # used to be PositiveDefinite
    ])

    # 2. Define the cost as a standard Python function
    # The decorator tells pymanopt to use autograd to compute gradients automatically
    # @pymanopt.function.autograd(manifold)
    @pymanopt.function.pytorch(manifold)
    def cost(U1, U2, B):
        """
        Defines the cost function using standard NumPy-like operations.
        The variables U1, U2, and B will be passed by the solver.
        """
        U1 = U1.to(torch.float32)
        U2 = U2.to(torch.float32)
        B = B.to(torch.float32)

        # 2. Move pre-computed matrices to the target torch device
        sXtX = torch.from_numpy(XtX).to(B.device, dtype=torch.float32)
        sZtZ = torch.from_numpy(ZtZ).to(B.device, dtype=torch.float32)
        sXtAZ = torch.from_numpy(XtAZ).to(B.device, dtype=torch.float32)

        # Combine U1, U2, B to form the transformation matrix W
        W = (U1 @ B) @ U2.T

        # Regularization term (same as before)
        regularization_term = 0.5 * Lambda * torch.sum(B ** 2)

        # First part of the cost: Tr(W^T * (X^T*X) * W * (Z^T*Z))
        # This is the expanded reconstruction error term
        trace_term = torch.trace((W.T @ sXtX @ W) @ sZtZ)

        # Second part of the cost: -2 * sum(W * (X^T*A*Z))
        # This is the expanded cross-term between reconstructed and original A
        correlation_term = -2 * torch.sum(W * sXtAZ)

        return regularization_term + trace_term + correlation_term

    # 3. Instantiate the pymanopt problem
    # We pass the manifold and the cost function directly
    print(f'Before pymanopt.Problem')
    problem = pymanopt.Problem(manifold=manifold, cost=cost)

    # 4. Define and run the solver (this part is unchanged)
    print(f'Before ConjugateGradient')
    solver = ConjugateGradient(max_time=15000, max_iterations=150)  # max_iterations=args.max_opt_iter)
    wopt = solver.run(problem)

    # 5. Unpack the optimized weights
    U1, U2, B = wopt.point

    # Step 2: Transformation
    sqrtm_B = xp.asarray(scipy.linalg.sqrtm(B), dtype=dtype)
    U1 = xp.asarray(U1, dtype=dtype)
    U2 = xp.asarray(U2, dtype=dtype)

    result_xw2 = U1.dot(sqrtm_B)
    result_zw2 = U2.dot(sqrtm_B)

    result_xw = x.dot(U1).dot(sqrtm_B)
    result_zw = z.dot(U2).dot(sqrtm_B)

    end = time.time()
    print(f"Geomm runtime: {end - start} seconds")

    return result_xw, result_zw, result_xw2, result_zw2

if __name__ == '__main__':
    main()
