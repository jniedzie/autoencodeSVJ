//
//  Jet.hpp
//  h5parser
//
//  Created by Jeremi Niedziela on 17/07/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef Jet_hpp
#define Jet_hpp

#include "Helpers.hpp"
#include "Constituent.hpp"

class Jet
{
public:
  Jet(){}
  
  
  double eta, phi, pt, mass, chargedFraction, PTD, axis2, flavor, energy;
  vector<double> EFPs;
  vector<shared_ptr<Constituent>> constituents;
  
  void print();
};

#endif /* Jet_hpp */
