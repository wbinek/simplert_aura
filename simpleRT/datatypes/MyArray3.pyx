cimport cython
cimport numpy as np
from cpython.array cimport array, clone

from simpleRT.datatypes.MyVec3 cimport MyVec3

cdef array template
global template
template = array('f')

@cython.final
cdef class MyArray3():
    def __cinit__(self, MyArray3 other = None):
        global template
        self._cdata = clone(template, 9, False)
        #self._mdata = self._cdata
        #self._data = <float[:3,:3]> &self._mdata[0]

        if other:
            self.set_matrix(other[0,0], other[0,1], other[0,2],\
                            other[1,0], other[1,1], other[1,2],\
                            other[2,0], other[2,1], other[2,2])

    #def __getitem__(self,tuple index):
    #    cdef int i,j
    #    i,j = index[0],index[1]
    #    return self._data[i,j]

    def __str__(self):
        l = self.asList()
        return str(l)

    def __neg__(self):
        res = MyArray3()
        cdef int i
        for i in range(9):
            res._cdata.data.as_floats[i] = -self._cdata.data.as_floats[i]
        return res

    cdef set_zeros(self):
        cdef int i
        for i in range(9):
                self._cdata.data.as_floats[i]=0.

    cdef set_matrix(self,float x11,float x12,float x13,float x21,float x22,float x23,float x31,float x32,float x33):
        self._cdata.data.as_floats[0] = x11 
        self._cdata.data.as_floats[1] = x12
        self._cdata.data.as_floats[2] = x13
        self._cdata.data.as_floats[3] = x21
        self._cdata.data.as_floats[4] = x22
        self._cdata.data.as_floats[5] = x23
        self._cdata.data.as_floats[6] = x31
        self._cdata.data.as_floats[7] = x32
        self._cdata.data.as_floats[8] = x33

    cdef list asList(self):
        return [[self._cdata.data.as_floats[0],self._cdata.data.as_floats[1],self._cdata.data.as_floats[2]],\
                [self._cdata.data.as_floats[3],self._cdata.data.as_floats[4],self._cdata.data.as_floats[5]],\
                [self._cdata.data.as_floats[6],self._cdata.data.as_floats[7],self._cdata.data.as_floats[8]]]        

    cdef MyArray3 eye(self):
        self.set_zeros()
        self._cdata.data.as_floats[0] = 1.
        self._cdata.data.as_floats[4] = 1.
        self._cdata.data.as_floats[8] = 1.

    cdef MyArray3 add_array(self, MyArray3 a2):
        res = MyArray3()
        cdef int i
        for i in range(9):
                res._cdata.data.as_floats[i] = self._cdata.data.as_floats[i]+a2._cdata.data.as_floats[i]
        return res

    cdef MyArray3 add_float(self, float v):
        res = MyArray3()
        cdef int i
        for i in range(9):
            res._cdata.data.as_floats[i] = self._cdata.data.as_floats[i]+v
        return res

    cdef MyArray3 mul_float(self, float v):
        res = MyArray3()
        cdef int i
        for i in range(9):
            res._cdata.data.as_floats[i] = self._cdata.data.as_floats[i]*v
        return res

    cdef MyVec3 mul_vec(self, MyVec3 v):
        res = MyVec3()
        res.x = self._cdata.data.as_floats[0]*v.x + self._cdata.data.as_floats[1]*v.y + self._cdata.data.as_floats[2]*v.z
        res.y = self._cdata.data.as_floats[3]*v.x + self._cdata.data.as_floats[4]*v.y + self._cdata.data.as_floats[5]*v.z
        res.z = self._cdata.data.as_floats[6]*v.x + self._cdata.data.as_floats[7]*v.y + self._cdata.data.as_floats[8]*v.z
        return res

    cdef MyArray3 mul_array(self, MyArray3 v):
        res = MyArray3()
        res._cdata.data.as_floats[0] = self._cdata.data.as_floats[0]*v._cdata.data.as_floats[0] + \
                                        self._cdata.data.as_floats[1]*v._cdata.data.as_floats[3] + \
                                        self._cdata.data.as_floats[2]*v._cdata.data.as_floats[6]

        res._cdata.data.as_floats[1] = self._cdata.data.as_floats[0]*v._cdata.data.as_floats[1] + \
                                        self._cdata.data.as_floats[1]*v._cdata.data.as_floats[4] + \
                                        self._cdata.data.as_floats[2]*v._cdata.data.as_floats[7]

        res._cdata.data.as_floats[2] = self._cdata.data.as_floats[0]*v._cdata.data.as_floats[2] + \
                                        self._cdata.data.as_floats[1]*v._cdata.data.as_floats[5] + \
                                        self._cdata.data.as_floats[2]*v._cdata.data.as_floats[8]

        res._cdata.data.as_floats[3] = self._cdata.data.as_floats[3]*v._cdata.data.as_floats[0] + \
                                        self._cdata.data.as_floats[4]*v._cdata.data.as_floats[3] + \
                                        self._cdata.data.as_floats[5]*v._cdata.data.as_floats[6]

        res._cdata.data.as_floats[4] = self._cdata.data.as_floats[3]*v._cdata.data.as_floats[1] + \
                                        self._cdata.data.as_floats[4]*v._cdata.data.as_floats[4] + \
                                        self._cdata.data.as_floats[5]*v._cdata.data.as_floats[7]

        res._cdata.data.as_floats[5] = self._cdata.data.as_floats[3]*v._cdata.data.as_floats[2] + \
                                        self._cdata.data.as_floats[4]*v._cdata.data.as_floats[5] + \
                                        self._cdata.data.as_floats[5]*v._cdata.data.as_floats[8]

        res._cdata.data.as_floats[6] = self._cdata.data.as_floats[6]*v._cdata.data.as_floats[0] + \
                                        self._cdata.data.as_floats[7]*v._cdata.data.as_floats[3] + \
                                        self._cdata.data.as_floats[8]*v._cdata.data.as_floats[6]

        res._cdata.data.as_floats[7] = self._cdata.data.as_floats[6]*v._cdata.data.as_floats[1] + \
                                        self._cdata.data.as_floats[7]*v._cdata.data.as_floats[4] + \
                                        self._cdata.data.as_floats[8]*v._cdata.data.as_floats[7]

        res._cdata.data.as_floats[8] = self._cdata.data.as_floats[6]*v._cdata.data.as_floats[2] + \
                                        self._cdata.data.as_floats[7]*v._cdata.data.as_floats[5] + \
                                        self._cdata.data.as_floats[8]*v._cdata.data.as_floats[8]
        return res


    cdef MyArray3 square(self):
        res = MyArray3()
        cdef int i
        for i in range(9):
            res._cdata.data.as_floats[i] = self._cdata.data.as_floats[i]*self._cdata.data.as_floats[i]
        return res


