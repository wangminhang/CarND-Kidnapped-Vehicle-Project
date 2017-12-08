/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <float.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 150;

    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_th(theta, std[2]);

    for (int m = 0; m < num_particles; m++)
    {
        Particle particle;
        particle.id = m;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_th(gen);
        particle.weight = 1;
        particles.push_back(particle);
        weights.push_back(1);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;

    for (auto& p : particles)
    {
        double new_x, new_y, new_theta;
        if(yaw_rate == 0)
        {
            new_x = p.x + velocity * delta_t * cos(p.theta);
            new_y = p.y + velocity * delta_t * sin(p.theta);
            new_theta = p.theta;
        }
        else
        {
            new_x = p.x + velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            new_y = p.y + velocity/yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
            new_theta = p.theta + yaw_rate * delta_t;
        }

        normal_distribution<double> dist_x(new_x, std_pos[0]);
        normal_distribution<double> dist_y(new_y, std_pos[1]);
        normal_distribution<double> dist_th(new_theta, std_pos[2]);

        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_th(gen);
    }
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    vector<LandmarkObs> associated_landmarks;
    LandmarkObs nearest_landmark;
    double shortest_distance;

    for(auto& obs: observations)
    {
        shortest_distance = DBL_MAX; // max double value
        nearest_landmark = obs; // init with the first observed landmark
        for(auto pred: predicted)
        {
            double distance = dist(obs.x, obs.y, pred.x, pred.y);
            if (distance < shortest_distance)
            {
                shortest_distance = distance;
                nearest_landmark = pred;
            }
        }
        associated_landmarks.push_back(nearest_landmark);
    }
    return associated_landmarks;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    const double sigma_x = std_landmark[0];
    const double sigma_y = std_landmark[1];
    const double sigma_xy_2_pi = 2 * M_PI * sigma_x * sigma_y;
    const double sigma_xx_2 = 2 * sigma_x * sigma_x;
    const double sigma_yy_2 = 2 * sigma_y * sigma_y;

    for(int m = 0; m < num_particles; m++)
    {
        Particle p = particles[m];

        vector<LandmarkObs> trans_observations;
        for(auto obs: observations)
        {
            LandmarkObs trans_obs;
            trans_obs.id = p.id;
            trans_obs.x = p.x + (obs.x * cos(p.theta) - obs.y * sin(p.theta));
            trans_obs.y = p.y + (obs.x * sin(p.theta) + obs.y * cos(p.theta));
            trans_observations.push_back(trans_obs);
        }

        vector<LandmarkObs> predictions;
        for(auto landmark: map_landmarks.landmark_list)
        {
            double distance = dist(p.x, p.y, landmark.x_f, landmark.y_f);
            if(distance < sensor_range)
            {
                LandmarkObs lm_tmp;
                lm_tmp.x = landmark.x_f;
                lm_tmp.y = landmark.y_f;
                lm_tmp.id = landmark.id_i;
                predictions.push_back(lm_tmp);
            }
        }

        auto associated_landmarks = dataAssociation(predictions, trans_observations);

        double prob = 1.0;
        for (int n = 0; n < associated_landmarks.size(); n++)
        {
            double dx = trans_observations.at(n).x - associated_landmarks.at(n).x;
            double dy = trans_observations.at(n).y - associated_landmarks.at(n).y;
            prob *= 1.0/(sigma_xy_2_pi)*exp(-dx*dx/(sigma_xx_2))*exp(-dy*dy/(sigma_yy_2));
        }

        weights[m] = prob;
    }
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    vector<Particle> resample_particles;
    discrete_distribution<int> distribution(weights.begin(), weights.end());

    for(int i=0; i < num_particles; i++){
        resample_particles.push_back(particles[distribution(gen)]);
    }
    particles = resample_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
