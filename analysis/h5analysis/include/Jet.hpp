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
  double ecf1, ecf2, ecf3, e2, C2, D3;
  vector<double> EFPs;
  vector<shared_ptr<Constituent>> constituents;
  
  bool isEmpty();
  
  void print();
};

#endif /* Jet_hpp */
