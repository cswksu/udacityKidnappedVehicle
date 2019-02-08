/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;
  double tempX;
  double tempY;
  double tempTheta;
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  Particle tempP;
  for (int i = 0; i < num_particles; i++) {
    tempX = dist_x(gen);
    tempY = dist_y(gen);
    tempTheta = dist_theta(gen);
    tempP.x = tempX;
    tempP.y = tempY;
    tempP.theta = tempTheta;
    tempP.weight = 1;
    tempP.id = i;
    particles.push_back(tempP);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i = 0; i < num_particles; i++) {
    particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta))+dist_x(gen);
    particles[i].y += velocity / yaw_rate * (cos(particles[i].theta)-cos(particles[i].theta+yaw_rate*delta_t));
    particles[i].theta += yaw_rate * delta_t;
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double tempMinDist = 999;
  int tempMinIdx = -1;
  double compareDist;
  // for each observation
  for (int i = 0; i < observations.size(); i++) {
    //loop through all predictions to find closest prediction
    tempMinDist = 999;
    tempMinIdx = -1;
    for (int j = 0; j < predicted.size(); j++) {
      compareDist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (compareDist < tempMinDist) {
        tempMinDist = compareDist;
        tempMinIdx = j;
      }


    }
    observations[i].id = predicted[tempMinIdx].id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  double xm;
  double ym;
  vector<LandmarkObs> transObs;
  LandmarkObs tempTransObs;
  vector<LandmarkObs> pred;
  LandmarkObs tempPred;
  double tempDist;
  double w_prefix = 1 / (2 * M_PI*std_landmark[0] * std_landmark[1]);
  double tempWeight;
  for (int i = 0; i < num_particles; i++) {
    pred.clear();
    transObs.clear();
    tempWeight = 1;
    for (int j = 0; j < observations.size(); j++) {
      xm = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
      ym = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
      tempTransObs.x = xm;
      tempTransObs.y = ym;
      tempTransObs.id = observations[j].id;
      transObs.push_back(tempTransObs);
    }
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      tempDist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if (tempDist < sensor_range) {
        tempPred.id = map_landmarks.landmark_list[j].id_i;
        tempPred.x = map_landmarks.landmark_list[j].x_f;
        tempPred.y = map_landmarks.landmark_list[j].y_f;
        pred.push_back(tempPred);

      }
    }
    dataAssociation(pred, transObs);
    for (int j = 0; j < observations.size(); j++) {
      tempWeight *= w_prefix * (pow(observations[j].x - map_landmarks.landmark_list[observations[j].id].x_f, 2) / (2 * pow(std_landmark[0], 2)) + pow(observations[j].y - map_landmarks.landmark_list[observations[j].id].y_f, 2) / (2 * pow(std_landmark[1], 2)));
    }
    particles[i].weight = tempWeight;
  }
  
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<double> idxWeights;
  for (int i = 0; i < num_particles; i++) {
    idxWeights.push_back(particles[i].weight);
  }
  std::default_random_engine generator;
  double a[num_particles];
  std::copy(idxWeights.begin(), idxWeights.end(), a);
  std::discrete_distribution<> distrib(a);
  vector<Particle> resampledParticles;
  for (int i = 0; i < num_particles; i++) {
    resampledParticles.push_back(particles[distrib(generator)]);
  }
  for (int i = 0; i < num_particles; i++) {
    particles[i]=resampledParticles[i];
  }
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}