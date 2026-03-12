import math
import numpy as np
import krpc

#Represents the state of the current vessle in KRP. Contains all inputs
#to be used in Deep-Q approach.
class KSPState:
    def __init__(self, target_lat, target_lon, conn_name="Craft"):
        self.conn = krpc.connect(name=conn_name)
        self.sc = self.conn.space_center
        self.vessel = self.sc.active_vessel
        self.flight = self.vessel.flight()
        self.control = self.vessel.control
        self.resources = self.vessel.resources

        self.target_lat = target_lat
        self.target_lon = target_lon
        self.body = self.vessel.orbit.body #planet (usualy kerbin which is the earth analoge)

    #degree to radian
    def _deg2rad(self, a):
        return a * math.pi / 180.0
    
    #wrap angle to -180 <= angle >= 180
    def _wrap_angle_deg(self, angle):
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    #distance to target along a sphere given only latitude and longitudes. See https://en.wikipedia.org/wiki/Haversine_formula
    def _haversine_distance(self, lat1, lon1, lat2, lon2, r):
        lat1 = self._deg2rad(lat1)
        lat2 = self._deg2rad(lat2)
        lon1 = self._deg2rad(lon1)
        lon2 = self._deg2rad(lon2)

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1

        phi = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
            )
        
        theta = 2 * math.atan2(math.sqrt(phi), math.sqrt(1 - phi))
        return theta * r
    
    #angle between two points on a sphere using lat and lon
    #This is the initial bearing, note that this will change at each point along a sphereical curve
    #So this will be called the forward azimuth
    #see https://www.movable-type.co.uk/scripts/latlong.html
    def _angle_to_target(self, lat1, lon1, lat2, lon2):
        lat1 = self._deg2rad(lat1)
        lat2 = self._deg2rad(lat2)
        lon1 = self._deg2rad(lon1)
        lon2 = self._deg2rad(lon2)

        delta_lon = lon2 - lon1

        y = math.sin(delta_lon) * math.cos(lat2)
        x = (
            math.cos(lat1) * math.sin(lat2)
            - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        )

        b = math.degrees(math.atan2(y, x))
        return (b + 360) % 360
    
    #normalize to a given value, note that due to the nature of KSP I cannot garantue [-1, 1]
    def _normalize(self, value, usual_scale):
        return max(-1.0, min(1.0, value / usual_scale))
    

    def distance_to_target(self):
        return self._haversine_distance(
            self.flight.latitude,
            self.flight.longitude,
            self.target_lat,
            self.target_lon,
            self.body.equatorial_radius
        )
    
    def angle_error(self):
        target = self._angle_to_target(
            self.flight.latitude,
            self.flight.longitude,
            self.target_lat,
            self.target_lon,
        )

        current = self.flight.heading
        return self._wrap_angle_deg(target - current)
    
    def fuel(self):
        try:
            max_fuel = self.resources.max("LiquidFuel") #get max amount of fuel in vessel
            if max_fuel <= 0:
                return 1.0
            return self.resources.amount("LiquidFuel") / max_fuel
        except Exception:
            return 1.0
        
    def get_state_as_dict(self):
        return {
            "distance_to_target": self.distance_to_target(),
            "heading_error": self.angle_error(),
            "speed": self.flight.speed,
            "vertical_speed": self.flight.vertical_speed,
            "pitch": self.flight.pitch,
            "roll": self.flight.roll,
            "heading": self.flight.heading,
            "angle_of_attack": self.flight.angle_of_attack,
            "fuel_fraction": self.fuel(),
            "throttle": self.control.throttle,
        }
    
    #state vector with normalized
    def get_state_as_vector(self):
        state = self.get_state_as_dict()

        return np.array([
            self._normalize(state["distance_to_target"], 100000.0),
            self._normalize(state["heading_error"], 180.0),
            self._normalize(state["speed"], 300.0),
            self._normalize(state["vertical_speed"], 100.0),
            self._normalize(state["pitch"], 90.0),
            self._normalize(state["roll"], 180.0),
            self._normalize(state["heading"], 180.0),
            self._normalize(state["angle_of_attack"], 30.0),
            max(0.0, min(1.0, state["fuel_fraction"])),
            max(0.0, min(1.0, state["throttle"])),
        ], dtype=np.float32)
    
    def close(self):
        self.conn.close()