

# Duckietown World


This package provides functions to reason about the state 
of the world in Duckietown.



## Coordinates

Each coordinate specification has two attributes:

1. The frame to which it refers to; represented as a string.
2. The "parametrization"; represented by the class type.


Example reference frames:

    tile            current tile
    lane            current lane center
    intersection    current intersection
    camera          camera-centered frame
    egovehicle      vehicle-center frame
    
Example parametrization:

    SE2
        x
        y
        theta
    
    R2  
        x
        y
    
    LanePosition:
        angle
        dist_from_center
        
    Bearing2D
    
    Bearing3D
    
    Distance2D
  
    Distance3D
  
    SignedDistance2D
    
For example:

* The distance of the Duckiebot from the stop line is a SignedDistance


# Time

Each quantity can be queried at different times, in the past and in the future. 

    pos = d
    
    pos_distribution = ego.position.wrt('tile').at(NOW)  
 
# Uncertainty

Each information returned is a distribution.

    p_tile = ego.position.wrt('tile').at(NOW)

You can obtain:

    mle = p.mle()
    
    p.samples()
    p.coeff()
    
    
   

# Objects

    Each object has a position specification
    
    
    

# Class `EgoVehicle`


    
# Simple operations


    dw = DuckietownWorld.from_yaml()
    
    
    ego = dw.get_ego_vehicle()
