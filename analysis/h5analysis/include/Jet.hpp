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
  
  
  double eta, phi, pt, mass, chargedFraction, PTD, axisMinor, axisMajor, girth, lha, flavor, energy;
  double e2, e3, C2, D2;
  vector<double> EFPs;
  vector<shared_ptr<Constituent>> constituents;
  
  bool isEmpty();
  
  void print();
};

#endif /* Jet_hpp */
