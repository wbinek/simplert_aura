cimport cython
cimport numpy as np

from libc.math cimport sqrt

## Vector class
@cython.final
@cython.freelist(8)
cdef class MyVec3:

    def __str__(self):
        l = self.asArray()
        return str(l)

    def __neg__(self):
        cdef MyVec3 res = MyVec3()
        res.x, res.y, res.z = -self.x, -self.y, -self.z
        return res
            
    cdef  float dot(self, MyVec3 v2):
        return self.x*v2.x + self.y*v2.y + self.z*v2.z
    
    cdef MyVec3 cross(self, MyVec3 v2):
        cdef MyVec3 cross = MyVec3()
        cross.x = self.y*v2.z - self.z*v2.y
        cross.y = self.z*v2.x - self.x*v2.z
        cross.z = self.x*v2.y - self.y*v2.x      
        return cross

    cdef inline float vectorLength(self):
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    cdef MyVec3 add_vec(self, MyVec3 v2):
        cdef MyVec3 added = MyVec3()
        added.x = self.x + v2.x
        added.y = self.y + v2.y
        added.z = self.z + v2.z
        return added

    cdef MyVec3 add_float(self, float v2):
        cdef MyVec3 added = MyVec3()
        added.x = self.x + v2
        added.y = self.y + v2
        added.z = self.z + v2
        return added

    cdef MyVec3 sub_vec(self,MyVec3 v2):      
        cdef MyVec3 subs = MyVec3()
        subs.x = self.x - v2.x
        subs.y = self.y - v2.y
        subs.z = self.z - v2.z
        return subs

    cdef MyVec3 sub_float(self,float v2): 
        cdef MyVec3 subs = MyVec3()
        subs.x = self.x - v2
        subs.y = self.y - v2
        subs.z = self.z - v2
        return subs

    cdef MyVec3 mul(self, float a):
        cdef MyVec3 multi = MyVec3()
        multi.x = self.x * a
        multi.y = self.y * a
        multi.z = self.z * a
        return multi

    cdef MyVec3 div(self, float a):
        cdef MyVec3 divi = MyVec3()
        divi.x = self.x / a
        divi.y = self.y / a
        divi.z = self.z / a
        return divi

    cdef MyVec3 neg(self):
        cdef MyVec3 nega = MyVec3()
        nega.x = -self.x
        nega.y = -self.y
        nega.z = -self.z
        return nega
    
    cdef MyVec3 normalize(self):
        cdef float length = self.vectorLength()
        cdef MyVec3 normalized = self.div(length)
        return normalized
    
    cpdef list asArray(self):
        return [self.x, self.y, self.z]
    
    cpdef fromArray(self, list vec):
        self.x, self.y, self.z = vec[0], vec[1], vec[2]