//
//  Event.hpp
//  h5parser
//
//  Created by Jeremi Niedziela on 17/07/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#ifndef Event_hpp
#define Event_hpp

#include "Helpers.hpp"
#include "Jet.hpp"

class Event
{
public:
  Event(){}
  
  double MET, METeta, METphi, MT, Mjj;
  vector<shared_ptr<Jet>> jets;
  
  void print();
};

#endif /* Event_hpp */
