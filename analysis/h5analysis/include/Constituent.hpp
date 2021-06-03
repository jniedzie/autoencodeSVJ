//
//  Constituent.hpp
//  h5analysis
//
//  Created by Jeremi Niedziela on 21/04/2021.
//  Copyright © 2021 Jeremi Niedziela. All rights reserved.
//

#ifndef Constituent_hpp
#define Constituent_hpp

#include "Helpers.hpp"

class Constituent
{
public:
  Constituent(){}
  
  
  double eta, phi, pt, energy, rapidity;
  
  void print();
  
  bool isEmpty();
  
};

#endif /* Constituent_hpp */
