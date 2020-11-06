import cython
from cpython.array cimport array

from simpleRT.datatypes.MyVec3 cimport MyVec3

@cython.final
cdef class MyArray3():
    cdef array _cdata
    #cdef float[:] _mdata
    #cdef float[:,:] _data

    cdef void set_zeros(self)

    cdef void set_matrix(self,float x11,float x12,float x13,float x21,float x22,float x23,float x31,float x32,float x33)

    cdef list asList(self)

    cdef MyArray3 eye(self)

    cdef MyArray3 add_array(self, MyArray3 a2)

    cdef MyArray3 add_float(self, float v)

    cdef MyArray3 mul_float(self, float v)

    cdef MyVec3 mul_vec(self, MyVec3 v) 

    cdef MyArray3 mul_array(self, MyArray3 v)

    cdef MyArray3 square(self)


