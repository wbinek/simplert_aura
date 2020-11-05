import cython
from cpython cimport bool

cdef class MyVec3:
    cdef float x;
    cdef float y;
    cdef float z;
    
    cdef float dot(self, MyVec3 v2)
	
    cdef MyVec3 cross(self, MyVec3 v2)

    cdef float vectorLength(self)

    cdef MyVec3 add_vec(self, MyVec3 v2)

    cdef MyVec3 add_float(self, float v2)

    cdef MyVec3 sub_vec(self,MyVec3 v2)

    cdef MyVec3 sub_float(self,float v2)

    cdef MyVec3 mul(self, float a)

    cdef MyVec3 div(self, float a)

    cdef MyVec3 neg(self)
    
    cdef MyVec3 normalize(self)

    cdef MyVec3 minv(self, MyVec3 v2)

    cdef MyVec3 maxv(self, MyVec3 v2)
    
    cpdef list asArray(self)
    
    cpdef fromArray(self, list vec)
	