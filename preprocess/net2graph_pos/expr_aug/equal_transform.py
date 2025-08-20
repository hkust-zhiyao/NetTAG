from pysmt.shortcuts import Symbol, And, Or, Not, Implies, Iff, is_sat, Ite, Xor, Plus, Equals, Times, Real, GE, LT, LE, GT, Minus, EqualsOrIff, TRUE, FALSE
from pysmt.typing import BOOL
import random

## double negation
## Rule: Not(Not(A)) ≡ A
def double_negation(expr):
    if expr.is_not() and expr.arg(0).is_not():
        return expr.arg(0).arg(0)
    return expr

## De Morgan's laws
## Rule: Not(And(A, B)) ≡ Or(Not(A), Not(B))
def de_morgan(expr):
    if expr.is_not():
        arg = expr.arg(0)
        if arg.is_and():
            return Or([Not(sub) for sub in arg.args()])
        elif arg.is_or():
            return And([Not(sub) for sub in arg.args()])
    return expr

## Distributive laws
## Rule: And(A, Or(B, C)) ≡ Or(And(A, B), And(A, C))
def distributive(expr):
    if expr.is_and():
        args = expr.args()
        for arg in args:
            if arg.is_or():
                other_args = [a for a in args if a != arg]
                new_args = [And([sub_arg] + other_args) for sub_arg in arg.args()]
                return Or(new_args)
    elif expr.is_or():
        args = expr.args()
        for arg in args:
            if arg.is_and():
                other_args = [a for a in args if a != arg]
                new_args = [Or([sub_arg] + other_args) for sub_arg in arg.args()]
                return And(new_args)
    return expr

## Commutative Laws
## Rule: And(A, B) ≡ And(B, A)
def commutative(expr):
    if expr.is_and() or expr.is_or():
        args = list(expr.args())
        random.shuffle(args)
        return expr.node_type()(args)
    return expr

## Associative Laws
## Rule: And(A, And(B, C)) ≡ And(And(A, B), C)
def associative(expr):
    if expr.is_and():
        new_args = []
        for arg in expr.args():
            if arg.is_and():
                new_args.extend(arg.args())
            else:
                new_args.append(arg)
        return And(new_args)
    elif expr.is_or():
        new_args = []
        for arg in expr.args():
            if arg.is_or():
                new_args.extend(arg.args())
            else:
                new_args.append(arg)
        return Or(new_args)
    return expr

## Idempotent Laws
## Rule: And(A, A) ≡ A
def idempotent(expr):
    if expr.is_and() or expr.is_or():
        args = set(expr.args())
        if len(args) < len(expr.args()):
            return expr.node_type()(list(args))
    return expr

## Absorption Laws
## Rule: And(A, Or(A, B)) ≡ A
def absorption(expr):
    if expr.is_or():
        for arg in expr.args():
            if arg.is_and():
                and_args = arg.args()
                for sub_arg in and_args:
                    if sub_arg in expr.args():
                        return sub_arg
    elif expr.is_and():
        for arg in expr.args():
            if arg.is_or():
                or_args = arg.args()
                for sub_arg in or_args:
                    if sub_arg in expr.args():
                        return sub_arg
    return expr

## Negation Laws
## Rule: And(A, Not(A)) ≡ False
def negation_laws(expr):
    if expr.is_or():
        args = expr.args()
        for arg in args:
            if Not(arg) in args:
                return TRUE()
    elif expr.is_and():
        args = expr.args()
        for arg in args:
            if Not(arg) in args:
                return FALSE()
    return expr


def random_trans(expr):
    trans_rules = [
    double_negation,
    de_morgan,
    distributive,
    # commutative,
    associative,
    idempotent,
    absorption,
    negation_laws
    ]
    rule = random.choice(trans_rules)
    return rule(expr)

def apply_random_trans(expr, num_trans=1):
    for _ in range(num_trans):
        expr = random_trans(expr)
        # expr = simplify(expr)  # Simplify the expression after each transformation
    return expr