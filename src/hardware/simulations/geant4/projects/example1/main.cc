#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"

#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "Randomize.hh"

int main(int argc, char** argv)
{
    // Detect interactive mode (if no arguments) and define UI session
    G4UIExecutive* ui = nullptr;
    if (argc == 1) {
        ui = new G4UIExecutive(argc, argv);
    }

    // Choose the Random engine
    G4Random::setTheEngine(new CLHEP::RanecuEngine);

    // Construct the default run manager
    G4RunManager* runManager = new G4RunManager;

    // Set mandatory initialization classes
    runManager->SetUserInitialization(new DetectorConstruction());
    runManager->SetUserInitialization(new PhysicsList());
    runManager->SetUserAction(new PrimaryGeneratorAction());

    // Set user action classes
    RunAction* runAction = new RunAction();
    runManager->SetUserAction(runAction);

    EventAction* eventAction = new EventAction(runAction);
    runManager->SetUserAction(eventAction);

    SteppingAction* steppingAction = new SteppingAction(eventAction);
    runManager->SetUserAction(steppingAction);

    // Initialize visualization
    G4VisManager* visManager = new G4VisExecutive;
    visManager->Initialize();

    // Get the pointer to the User Interface manager
    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    // Process macro or start UI session
    if (!ui) {
        // batch mode
        G4String command = "/control/execute ";
        G4String fileName = argv[1];
        UImanager->ApplyCommand(command+fileName);
    } else {
        // interactive mode
        UImanager->ApplyCommand("/control/execute init.mac");
        ui->SessionStart();
        delete ui;
    }

    // Job termination
    delete visManager;
    delete runManager;

    return 0;
}
