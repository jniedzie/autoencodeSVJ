//
//  Event.cpp
//  h5parser
//
//  Created by Jeremi Niedziela on 17/07/2020.
//  Copyright © 2020 Jeremi Niedziela. All rights reserved.
//

#include "Event.hpp"

void Event::print()
{
  cout<<"Event:"<<endl;
  cout<<"\tMET: "<<MET<<"\tMETeta: "<<METeta<<"\tMETphi: "<<METphi;
  cout<<"\tMT: "<<MT<<"\tMjj: "<<Mjj<<endl;
  cout<<"\tgen weight: "<<genWeight<<endl;
  cout<<"\tJets:"<<endl;
  for(auto jet : jets) jet->print();
}
