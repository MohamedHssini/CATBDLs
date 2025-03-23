import tensorflow as tf
#jdsvk

Yager = "Yager"
Acal = "Acal"

zero = 1e-12


class YagerGenerator():

    def __init__(self, p):
        self.p = 1. * p

    def __call__(self, a):
        return tf.pow(1 - a, self.p)


class AcalGenerator():

    def __init__(self, p):
        self.p = p

    def __call__(self, a):
        return tf.pow(-tf.math.log(tf.clip_by_value(a, zero, 1. - zero)), self.p)


class Gcreate(object):

    def __init__(self):
        pass

    @staticmethod
    def getGen(type_generator: str):
        if type_generator == Yager:
            return YagerGenerator
        elif type_generator == Acal:
            return AcalGenerator
        else:
            raise Exception("Unknown Type for Generator Family")


class FuzzyOperator():

    def __init__(self, generator, type_generator: str):
        self.t = generator
        self.type_generator = type_generator

    def implication(self, a, b):
        type = self.type_generator
        if type == Yager:
            return tf.clip_by_value(self.t(b) - self.t(a), 0., 1.)
        elif type == Acal:
            return self.t(b) / (1 + self.t(a) * self.t(b))
        else:
            return Exception("Unknown Type for Generator Family")

class KnowledgeBase():
    def __init__(self):
        self.formulas = []

    def add(self, formula):
        self.formulas.append(formula)

    def loss(self,type):
        if type == Yager:
            return tf.reduce_sum(tf.clip_by_value(tf.add_n(self.formulas),0., 1.) ,axis=0)
        else:
            return tf.reduce_sum(tf.add_n(self.formulas), axis=0)
