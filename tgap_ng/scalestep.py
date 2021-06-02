from connection import connection
import math

class ScaleStep(object):
    """ Calculates map scale based on step of the generalization process. 
        It is used for steering generalization process. Some features should be generalized differently in different scale. 
        E.g.: river should merge at 1:10k but split at 1:100k. 
        
        It is based on formula from 'progressive data transfer' paper (Huang 2016). 
        
        :Note: In publication 'A matrix-based structure for vario-scale vector representation 
        over wide range of Map scales: the case of river network data'
            was another approach to calculate scale. It was more targeted for river network (lines). 
            It is based on constant for a smallest visible segment (0.4 mm).
    """
    def __init__(self, init_scale, topo_nm):
        self.sb = init_scale 
        # scale denominator of base/start map, e.g. 
        # Sb = 1000 for scale 1:1000 [-]
        # nb = total number of object on base map [-]
        #  d = total area of domain in m2 e.g. D = 1 000 000 m2 for 1x1 km domain [m2]
        self.nb, self.d  = self.from_db(topo_nm)
        # print((self.nb), self.d)
        
    def from_db(self, topo_nm):
        """ Retrieve the initial number of objects (faces), width as xmax-xmin and h as ymax-ymin"""
        with connection(False) as conn:
            sql = '''
            SELECT count(face_id),
                    st_xmax(st_extent(mbr_geometry)) - st_xmin(st_extent(mbr_geometry)), 
                    st_ymax(st_extent(mbr_geometry)) - st_ymin(st_extent(mbr_geometry))
                     FROM {0}_face;
            '''.format(topo_nm)        
            [init_objs, w, h] = conn.record(sql)
        # print((w, h))
        return float(init_objs), w * h

    def scale_for_step(self,  step):
        """
        Calculates the current scale denominator, given:

        * the current step number,
        * the base scale and 
        * the number of objects on the base map
        """
        nb = self.nb
        if step >= nb:
            return float('inf')
        try: # last step = zero devision
            scale = self.sb * math.sqrt(nb / (nb-step))
#            print( " sb := ", self.sb)
#            print( " nb := ", nb)
#            print( " stp:= ", step)
#            print( " n-s:= ", nb-step)
        except ZeroDivisionError:
            scale = float('inf')
        return scale



#    get_St_from_step(step) {
#        let Nb = this.tree.metadata.no_of_objects_Nb
#        let St = this.tree.metadata.start_scale_Sb * Math.pow(Nb / (Nb - step), 0.5)

#        //console.log('transform.js step, Nb, Sb, St:', step, Nb, this.tree.metadata.start_scale_Sb, St)
#        return St
#    }

    def step_for_scale(self, scale):
        reductionf = 1 - pow((self.sb / scale), 2)
        step = self.nb * reductionf
        return step

#        let reductionf = 1 - Math.pow(this.tree.metadata.start_scale_Sb / St, 2)
#        let step = this.tree.metadata.no_of_objects_Nb * reductionf 


    def density(self):
        """ Calculates map data density for initial scale"""
        nb = self.nb
        d = self.d
        sb = self.sb
        density = (nb / d)* sb**2
        print("lib_config.py, map density: ", density, "obj/m2")
        return density
    
#    def density_dynamic(self, no_obj, dominator):
#        """ Calculates map data density dynamically for given scale"""
#        nb =  no_obj*1.0
#        d = self.d
#        sb = dominator
#        density = (nb / d)* sb**2
#        print "lib_config.py, map density dynamic: ", density, "obj/m2"
#        return density

    @staticmethod
    def resolution_mpp(denominator, ppi = 96):
        """Real world size (resolution) for 1 pixel
        
        given:
            the scale denominator (denominator)
            a certain density of pixels on a screen (ppi)

        The resolution can be used to get a value for a threshold for
        line simplification.
        """
        # OGC ppi = 2.54 / 0.028
        inch_in_cm = 2.54 # cm
        pixel_in_cm = inch_in_cm / ppi
        pixel_in_m = pixel_in_cm / 100.0
        resolution_m_per_pixel = pixel_in_m * denominator
        return resolution_m_per_pixel


def _test():
    mapping = ScaleStep(10000, 'top10nl_drenthe')
    print(mapping)

    print((mapping.scale_for_step(0)))
    print((mapping.scale_for_step(10)))
    print((mapping.scale_for_step(100)))
    print((mapping.scale_for_step(1000)))
    print((mapping.scale_for_step(10000)))
    denom = mapping.scale_for_step(100000)
    eps = mapping.resolution_mpp(denom, 96)
    print(("1:{:.0f} -> {:.3f} ".format(denom, eps)))

    denominator = 750
    while denominator <= 12288000:
        res_mpp = mapping.resolution_mpp(denominator, 2.54/0.028)
        print((denominator, round(res_mpp, 3)))
        res_mpp = mapping.resolution_mpp(denominator, 96)
        print((denominator, round(res_mpp, 3)))
        denominator *= 2

if __name__ == "__main__":
    _test()
