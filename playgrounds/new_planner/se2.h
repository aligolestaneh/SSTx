#ifndef SE2_H
#define SE2_H

#include <vector>
#include <cmath>

// Simple SE2 Pose class for transformations
class SE2Pose {
public:
    double x, y, theta;
    
    SE2Pose(double x = 0.0, double y = 0.0, double theta = 0.0) 
        : x(x), y(y), theta(theta) {}
    
    // Create from vector [x, y, theta]
    SE2Pose(const std::vector<double>& pose) 
        : x(pose[0]), y(pose[1]), theta(pose[2]) {}
    
    // Composition operator: this_pose @ other_pose
    SE2Pose operator*(const SE2Pose& other) const {
        double cos_t = cos(theta);
        double sin_t = sin(theta);
        
        // Normalize the result angle to [-π, π]
        double new_theta = theta + other.theta;
        while (new_theta > M_PI) new_theta -= 2.0 * M_PI;
        while (new_theta < -M_PI) new_theta += 2.0 * M_PI;
        
        return SE2Pose(
            x + cos_t * other.x - sin_t * other.y,
            y + sin_t * other.x + cos_t * other.y,
            new_theta
        );
    }
    
    // Inverse pose
    SE2Pose inverse() const {
        double cos_t = cos(-theta);
        double sin_t = sin(-theta);
        
        // Normalize the inverse angle to [-π, π]
        double inv_theta = -theta;
        while (inv_theta > M_PI) inv_theta -= 2.0 * M_PI;
        while (inv_theta < -M_PI) inv_theta += 2.0 * M_PI;
        
        return SE2Pose(
            cos_t * (-x) - sin_t * (-y),
            sin_t * (-x) + cos_t * (-y),
            inv_theta
        );
    }
    
    // Difference: other_pose - this_pose (in local frame)
    SE2Pose difference(const SE2Pose& other) const {
        return inverse() * other;
    }
    
    // Distance to another pose (weighted Euclidean in SE(2))
    double distance(const SE2Pose& other, double angular_weight = 0.5) const {
        double distPos = positionDistance(other);
        double distTheta = angularDistance(other);
        return sqrt(distPos*distPos + angular_weight * distTheta*distTheta);
    }
    
    // Position-only distance (ignoring orientation)
    double positionDistance(const SE2Pose& other) const {
        double dx = x - other.x;
        double dy = y - other.y;
        return sqrt(dx*dx + dy*dy);
    }
    
    // Angular distance only (in radians, normalized to [0, π])
    double angularDistance(const SE2Pose& other) const {
        double dtheta = theta - other.theta;
        while (dtheta > M_PI) dtheta -= 2.0 * M_PI;
        while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
        return fabs(dtheta);
    }
    
    // To vector
    std::vector<double> toVector() const {
        return {x, y, theta};
    }
};

#endif // SE2_H 