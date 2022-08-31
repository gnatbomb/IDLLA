from pickle import GLOBAL
import numpy as np
import csv
import os
import glob
import shutil

# Retreives the log information for all play sessions, and outputs them into a single one-hot array.


# Retrieves log information associated with a single level being played.
def parse_timeline(filename, foldername, level_data):
    file = csv.reader(open('../MarioPCGStudy/AnonymizedDirectory/' + foldername + '/' + filename, newline=''))
    instance_events = list(file)

    # Used for calculating Mario's x position
    walk_right = 0
    run_right = 0

    # Removes events that are logged before the 0th time-step
    while int(instance_events[1][1]) < 0:
        instance_events.pop(1)

    n = range(1, len(instance_events))
    
    # Number of timesteps to include. (shortest play session in the set.)
    timesteps = 904

    # One-hot of events
    timeline = np.zeros((timesteps, event_count + level_data_count + 1), dtype=int)
    
    # One-hot for monitoring multi-timestep events, such as movement, mario state, or level data
    register = np.zeros((event_count + level_data_count), dtype=int)
    for i in range(level_data_count):
        register[event_count + i] = level_data[i]

    # Used to check how recently the register was used to update the timeline
    last_timestep = 0
    stoptime = 0

    # used for calculating mario x. Used integer and divides by 1000 (rounding down) when setting values to avoid using floating points.
    mariox = 2000
    walk = 429
    run = 857

    # For each timestep (interrupts when we reach timesteps)
    for i in n:
        event = instance_events[i]
        eventcode = events[event[0]]

        # End recording time steps if we have passed 904 timesteps
        if int(event[1]) > timesteps:
            # update timeline up to this point using register
            for step in range(last_timestep, timesteps):
                # Update events
                for evcode in range(event_count):
                    if register[evcode]:
                        timeline[step][evcode] = 1

                # Update Level data
                for evcode in range(level_data_count):
                    timeline[step][event_count + evcode] = register[event_count + evcode]

                # update mario x
                if timeline[step][4]:
                    if timeline[step][6]:
                        run_right += 1
                        mariox += run
                    else:
                        walk_right += 1
                        mariox += walk
                elif timeline[step][5]:
                    if timeline[step][6]:
                        run_right -= 1
                        mariox -= run
                    else:
                        walk_right -= 1
                        mariox -= walk
                    if mariox < 0:
                        mariox = 0

                # If mario died, reset mario's calculated x
                if timeline[step][0]:
                    run_right = 0
                    walk_right = 0
                    mariox = 2000

                timeline[step][-1] = mariox // 1000

            # Update the last time we used the register to now.
            last_timestep = int(event[1])

            # Signals to stop processing more time steps.
            stoptime = 1

        # Does one last register update
        if stoptime == 0:
            if int(event[1]) != last_timestep:
                # update timeline up to this point using register
                for step in range(last_timestep, int(event[1])):
                    # Update events from register
                    for evcode in range(event_count):
                        if register[evcode]:
                            timeline[step][evcode] = 1

                    # Update level data from register
                    for evcode in range(level_data_count):
                        timeline[step][event_count + evcode] = register[event_count + evcode]

                    # Calculate mario's X
                    if timeline[step][4]:
                        if timeline[step][6]:
                            run_right += 1
                            mariox += run
                        else:
                            walk_right += 1
                            mariox += walk
                    elif timeline[step][5]:
                        if timeline[step][6]:
                            run_right -= 1
                            mariox -= run
                        else:
                            walk_right -= 1
                            mariox -= walk

                    # If mario died, reset his calculated x position
                    if timeline[step][0]:
                        run_right = 0
                        walk_right = 0
                        mariox = 2000

                    # Record mario's x
                    timeline[step][-1] = mariox // 1000

                # Update the most recent use of the register to now.
                last_timestep = int(event[1])
            
            # Reset register on death
            if eventcode > 10 and eventcode < 16:
                for k in range(0, event_count):
                    register[k] = 0

            # Update register, or update timeline directly.
            if eventcode > 2 and eventcode < 11:
                # player state update
                if "Start" in event[0]:
                    register[eventcode] = 1
                else:
                    register[eventcode] = 0
            else:
                timeline[int(event[1])][eventcode] = 1

            # Update timeline from register
            for evcode in range(event_count):
                if register[evcode]:
                    timeline[-1][evcode] = 1

            # Update level data from register
            for evcode in range(level_data_count):
                timeline[-1][event_count + evcode] = register[event_count + evcode]

    return timeline
    

# Retrieves the "level data" as well as player demographic information for a level being played.
def get_level_data(subject_results, play_order, level_indices):
    level_name = subject_results[1 + play_order]
    level_data = ['Fun', 'Frustration', 'Challenge', 'Design', 'SMBRank', 'MarioRank', 'PlatformerRank', 'GamesRank', 'Level_ind']

    # Retrieve level rankings
    for i in range(0, 3):
        # get fun
        if (subject_results[4 + i] == level_name):
            level_data[0] = i

        # get frustration
        if (subject_results[7 + i] == level_name):
            level_data[1] = i

        # get challenge
        if (subject_results[10 + i] == level_name):
            level_data[2] = i

        # get design
        if (subject_results[13 + i] == level_name):
            level_data[3] = i

    # get SMBRank
    level_data[4] = subject_results[19]

    # get MarioRank
    level_data[5] = subject_results[20]

    # get PlatformerRank
    level_data[6] = subject_results[21]

    # get GamesRank
    level_data[7] = subject_results[22]

    # get level_ind
    level_data[8] = level_indices[level_name]

    return level_data
                

# Retrieves the logs associated with a player (represented by foldername).
def parse_subject(foldername, level_indices):
    timelines = os.listdir('../MarioPCGStudy/AnonymizedDirectory/' + foldername + '/')

    subject_results = results[results_indices[foldername]]

    folder_array = np.zeros((0, 41), dtype=int)

    level_num = 0
    for timeline in timelines:
        level_data = get_level_data(subject_results, level_num, level_indices)
        folder_array = np.concatenate((folder_array, parse_timeline(timeline, foldername, level_data)))
        level_num += 1

    return folder_array


# Retrieves and formats all log data
def parse_all():
    level_indices = {}
    level_index = 0

    # Figure out level play order for each player
    for player in range(1, len(results)):
        for lvl in range(2, 5):
            if results[player][lvl] not in level_indices.keys():
                level_indices[results[player][lvl]] = level_index
                level_index += 1

    output_array = np.zeros((0, 41), dtype=int)

    subjects = os.listdir('../MarioPCGStudy/AnonymizedDirectory/')

    # For each player, retrieve the logs associated with that player.
    for subject in subjects:
        if (subject[-1].isdigit()): # Ignore labels row
            output_array = np.concatenate((output_array, parse_subject(subject, level_indices)))

    # Save the final output array
    np.savetxt("MarioPCG/logs.csv", output_array, header=output_header, delimiter=",", fmt='%d')

    


# Dictionary for all events and their event codes.
events = {
    # level stuff
    'StartLevel': 0,
    'WonLevel': 1,
    'LostLevel': 2,

    # movement states
    'JumpStart': 3,
    'JumpEnd': 3,
    'RightMoveStart': 4,
    'RightMoveEnd': 4,
    'LeftMoveStart': 5,
    'LeftMoveEnd': 5,
    'RunStateStart': 6,
    'RunStateEnd': 6,
    'DuckStart': 7,
    'DuckEnd': 7,

    # character states
    'LittleStateStart': 8,
    'LittleStateEnd': 8,
    'LargeStateStart': 9,
    'LargeStateEnd': 9,
    'FireStateStart': 10,
    'FireStateEnd': 10,

    # Deaths
    'DieByGoomba': 11,
    'DeathByShell': 12,
    'DeathByBulletBill': 13,
    'DieByGreenKoopa': 14,
    'DeathByGap': 15,

    # One-offs
    'UnleashShell': 16,
    'BlockCoinDestroy': 17,
    'BlockPowerDestroy': 18,
    'FireKillGoomba': 19,
    'StompKillGoomba': 20,
    'StompKillGreenKoopa': 21,
    'ShellKillGoomba': 22,
    'ShellKillGreenKoopa': 23,
    'FireKillGreenKoopa': 24,
    'CollectCoin': 25,
    'BlockPowerDestroyBulletBill': 26,
    'StompKillBulletBill': 27,
    'ShellKillBulletBill': 28,
    'BlockCoinDestroyBulletBill': 29,
    'CollectCoinBulletBill': 30,
}

# Ordered list for putting the top row of the outputted csv.
output_header = "StartLevel, WonLevel, LostLevel, Jumping, Right_Move, Left_Move, Running, Ducking, Little, Large, Fire, DieByGoomba, DeathByShell, DeathByBulletBill, DieByGreenKoopa, DeathByGap, UnleashShell, BlockCoinDestroy, BlockPowerDestroy, FireKillGoomba, StompKillGoomba, StompKillGreenKoopa, ShellKillGoomba, ShellKillGreenKoopa, FireKillGreenKoopa, CollectCoin, BlockPowerDestroyBulletBill, StompKillBulletBill, ShellKillBulletBill, BlockCoinDestroyBulletBill, CollectCoinBulletBill, Fun, Frustration, Challenge, Design, SMBRank, MarioRank, PlatformerRank, GamesRank, levelname, mariox"

event_count = 31
level_data_count = 9

resultsfile = csv.reader(open('../MarioPCGStudy/AnonymizedDirectory/AnonResults.csv', newline=''))
results = list(resultsfile)

results_indices = {}

for i in range(1, len(results) ):
    results_indices[results[i][0]] = i

parse_all()