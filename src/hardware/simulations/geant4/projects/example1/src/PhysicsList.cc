#include "PhysicsList.hh"

#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4EmStandardPhysics.hh"
#include "G4EmStandardPhysics_option1.hh"
#include "G4EmStandardPhysics_option2.hh"
#include "G4EmStandardPhysics_option3.hh"
#include "G4EmStandardPhysics_option4.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4EmPenelopePhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4HadronPhysicsFTFP_BERT.hh"
#include "G4HadronPhysicsQGSP_BERT.hh"
#include "G4HadronInelasticQBBC.hh"
#include "G4HadronPhysicsINCLXX.hh"
#include "G4IonPhysics.hh"
#include "G4NeutronTrackingCut.hh"
#include "G4LossTableManager.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4ProcessManager.hh"

PhysicsList::PhysicsList()
: G4VModularPhysicsList(){
  SetVerboseLevel(1);

  // Default physics
  RegisterPhysics(new G4DecayPhysics());

  // EM physics
  RegisterPhysics(new G4EmStandardPhysics());

  // Hadron physics
  RegisterPhysics(new G4HadronElasticPhysics());
  RegisterPhysics(new G4HadronPhysicsFTFP_BERT());
  RegisterPhysics(new G4IonPhysics());
}

PhysicsList::~PhysicsList()
{
}

void PhysicsList::SetCuts()
{
  G4VUserPhysicsList::SetCuts();
}
