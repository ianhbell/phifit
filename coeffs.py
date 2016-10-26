from nose.tools import assert_raises

def ArrayDeconstructor(dims, x):
    o = []
    ileft = 0
    for dim in dims:
        if isinstance(dim, list):
            o2 = []
            for _dim in dim:
                o2.append(x[ileft:ileft+_dim])
                ileft += _dim
            o.append(o2)
        else:
            o.append(x[ileft:ileft+dim])
            ileft += dim
    return o

def test_array_element():
    assert_raises(ArrayDeconstructor([1], [3.4]))
    c = ArrayDeconstructor([1, [3, 2], 2], [1,2,3,4,5,6,7,8,9])
    assert(c == [[1],[[2,3,4],[5,6]],[7,8]])

if __name__=='__main__':
    import nose
    nose.runmodule()