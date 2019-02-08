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
  //initialize particles around x, y, theta coordinates with given std deviations
  
  num_particles = 100; //initialize with 100 particles
  std::default_random_engine gen;
  
  //generate guassian noise around coarse x, y, theta values
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  Particle tempP; //container for particle
  
  //create particle vector
  for (int i = 0; i < num_particles; i++) {
    //generate noisy x, y, theta
    tempP.x = dist_x(gen);
    tempP.y = dist_y(gen);
    tempP.theta = dist_theta(gen);
    tempP.weight = 1; //initial weights all 1
    tempP.id = i; //set dummy ID
    particles.push_back(tempP); //push back into array
  }
  is_initialized = true; //indicate that initialized

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  // prediction step
  std::default_random_engine gen;
  //generate prediction noise centered about 0
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);
  for (int i = 0; i < num_particles; i++) {
    if (yaw_rate != 0) { //bicycle model with nonzero yaw rate and noise
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_x(gen);
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + dist_y(gen);
      particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
    }
    else { //zero yaw rate case
      particles[i].x += velocity *delta_t * cos(particles[i].theta) + dist_x(gen);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
      particles[i].theta +=  dist_theta(gen);
    }
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  // for each observation, find closest landmark and associate landmark ID
  double tempMinDist = 999; //dummy values to trigger minimum on 1st run
  int tempMinIdx = -1;
  double compareDist; //temporary container to compare against current minimum
  // for each observation
  for (int i = 0; i < observations.size(); i++) {
    //loop through all predictions to find closest prediction
    tempMinDist = 999; //reset values
    tempMinIdx = -1;
    for (int j = 0; j < predicted.size(); j++) {
      //get distance to from current observation to current landmark
      compareDist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (compareDist < tempMinDist) { //if new minimum
        tempMinDist = compareDist; //store new minimum
        tempMinIdx = j; //store location of new minimum
      }


    }
    observations[i].id = predicted[tempMinIdx].id; //associate observation id with prediction id
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  //update weights for given particle based on z-score of associated landmarks
  vector<LandmarkObs> transObs; //vector of transformed observations
  LandmarkObs tempTransObs;
  vector<LandmarkObs> pred; //vector of predicted landmark locations
  LandmarkObs tempPred;
  double tempDist;
  double w_prefix = 1.0 / (2.0 * M_PI*std_landmark[0] * std_landmark[1]); //prefix to all weights
  double tempWeight; //temporary weight for particle
  int match_count; //number of landmarks in particle sensor range
  for (int i = 0; i < num_particles; i++) { //for all particles
    pred.clear(); //clear vectors for predictions, observations
    transObs.clear();
    tempWeight = 1; //set to unity as will be multiplied to create final weight
    for (int j = 0; j < observations.size(); j++) { //for all ovservations
      //translate observations to map coordinates
      //store in temporary observation object and push back into vector
      tempTransObs.x = cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y + particles[i].x;
      tempTransObs.y = sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y + particles[i].y;
      tempTransObs.id = observations[j].id;
      transObs.push_back(tempTransObs);
    }
    match_count = 0; //reset to 0
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) { //for all landmarks
      //get distance to particle
      tempDist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      if (tempDist < sensor_range) { //if in range
        tempPred.id = j; //store landmark index as ID
        tempPred.x = map_landmarks.landmark_list[j].x_f; //store x and y of landmark
        tempPred.y = map_landmarks.landmark_list[j].y_f;
        pred.push_back(tempPred); //push back into prediction vector
        match_count++; //increment number of matches
      }
    }
    
    if (match_count != observations.size()) { //check that landmarks in range are a match
      tempWeight = 0; //if not, kill particle
    }
    else { //if match
      dataAssociation(pred, transObs); //perform data association
      for (int j = 0; j < observations.size(); j++) { //for all observations
        //multiply temporary weight by new temporary weight
        tempWeight *= w_prefix * exp(-(pow(transObs[j].x - map_landmarks.landmark_list[transObs[j].id].x_f, 2) / (2 * pow(std_landmark[0], 2)) + pow(transObs[j].y - map_landmarks.landmark_list[transObs[j].id].y_f, 2) / (2 * pow(std_landmark[1], 2))));
      }
      
    }
    particles[i].weight = tempWeight; //store weight with particle
  }
  
}

void ParticleFilter::resample() {
  //resample according to weights
  vector<double> idxWeights; //create vector for index weights
  for (int i = 0; i < num_particles; i++) {
    idxWeights.push_back(particles[i].weight); //put weight in vector
  }
  std::default_random_engine generator;
  std::discrete_distribution<> distribution(idxWeights.begin(),idxWeights.end());
  //create discrete distribution using paricle weights
  
  vector<Particle> resampledParticles; //create vector for resampled particles
  for (int i = 0; i < num_particles; i++) {
    resampledParticles.push_back(particles[distribution(generator)]);
    //push back resampled particles
  }
  for (int i = 0; i < num_particles; i++) {
    particles[i]=resampledParticles[i];
    //replace particles with resampled particles
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
