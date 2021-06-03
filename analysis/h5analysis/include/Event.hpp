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
  double genWeight;
  vector<shared_ptr<Jet>> jets;
  
  double getSumJetPt();
  void print();
  
  double getPtWeight(TH1D *ptDistribution);
};

#endif /* Event_hpp */
