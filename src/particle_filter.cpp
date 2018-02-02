#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
num_particles = 120;
default_random_engine gen;
double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	 std_x = std[0];
	 std_y = std[1];
	 std_theta = std[2];
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	for (int i = 0; i < num_particles ; ++i) {
		 Particle p;
		 p.id = i;
		 p.x = dist_x(gen);
		 p.y = dist_y(gen);
		 p.theta = dist_theta(gen);
		 p.weight = 1;
		 particles.push_back(p);
		 weights.push_back(p.weight);
}
is_initialized = true;
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;
	double std_x, std_y, std_theta;
		 std_x = std_pos[0];
		 std_y = std_pos[1];
		 std_theta = std_pos[2];
		 normal_distribution<double> n_x(0, std_x);
		 normal_distribution<double> n_y(0, std_y);
		 normal_distribution<double> n_theta(0, std_theta);

for (int i = 0; i < num_particles ; ++i){
	if(fabs(yaw_rate)>=0.0001){
		particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t)- sin(particles[i].theta)) + n_x(gen);
		particles[i].y += (velocity/yaw_rate)*(-cos(particles[i].theta + yaw_rate*delta_t)+ cos(particles[i].theta))+ n_y(gen);
		particles[i].theta += yaw_rate*delta_t + n_theta(gen);
	}
	else{
		particles[i].x += velocity*cos(particles[i].theta)*delta_t + n_x(gen);
		particles[i].y += velocity*sin(particles[i].theta)*delta_t + n_y(gen);
		particles[i].theta = particles[i].theta + n_theta(gen);
	}
}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	for (int i = 0; i < num_particles ; ++i){
		particles[i].weight  =1;
		for(int j=0;j<observations.size();j++){
			double Xm, Ym;
			Xm= particles[i].x + cos(particles[i].theta)*observations[j].x - sin(particles[i].theta)*observations[j].y;
			Ym = particles[i].y + sin(particles[i].theta)*observations[j].x + cos(particles[i].theta)*observations[j].y;
			double distance_l=sensor_range, Xc=sensor_range,Yc=sensor_range;
			double distance;
			for(int k=0;k<map_landmarks.landmark_list.size();k++){
				distance = sqrt((map_landmarks.landmark_list[k].x_f-Xm)*(map_landmarks.landmark_list[k].x_f-Xm) + (map_landmarks.landmark_list[k].y_f-Ym)*(map_landmarks.landmark_list[k].y_f-Ym));
				if (distance< distance_l)
				{distance_l = distance;
					Xc = map_landmarks.landmark_list[k].x_f;
					Yc = map_landmarks.landmark_list[k].y_f;
				}
			}
			double sigmax, sigmay, dX, dY;
			dX = Xc-Xm;
			dY = Yc-Ym;
			sigmax = std_landmark[0];
			sigmay = std_landmark[1];
			particles[i].weight *= (1/(2*M_PI*sigmax*sigmay))*exp(-((dX*dX)/(2*sigmax*sigmax) + (dY*dY)/(2*sigmay*sigmay)));
		}
	weights[i] =particles[i].weight;
	}

}

void ParticleFilter::resample() {

	default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());
	std::vector<Particle> particlesN;
	for (int i = 0; i < num_particles ; ++i){
		int pos = distribution(gen);
		particlesN.push_back(particles[pos]);
	}
	particles = particlesN;
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
