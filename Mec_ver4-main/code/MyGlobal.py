class MyGlobals(object):
    folder_name = "Not changing/"
    
# def Run_FDQO(folder_name):
#     MyGlobals.folder_name = folder_name + '/'
#     try:
#         os.makedirs(RESULT_DIR + MyGlobals.folder_name)
#     except OSError as e:
#         print(e)
#     folder = RESULT_DIR + MyGlobals.folder_name
#     #self.reward_files = open(RESULT_DIR + MyGlobals.folder_name + "reward.csv","w")
#     FDQO_method = Model_Deep_Q_Learning(14,4)    #In model  size, action
#     baseline = None  # None if using FDQO, >0 and <1 if using baseline
#     threshold = 0.9     # if reward received bigger than threshold, using Fuzzy Logic
#     k = 0.6     # Same formula as BDQL
#     epsilon = 0.1
#     model = FDQO_method.build_model(epsilon = epsilon, name = i, file = folder,
#                                     k = k, threshold = threshold)
#     #Create enviroment FDQO
#     env = BusEnv("FDQO")
#     env.modifyEnv(i, folder)
#     env.seed(123)
#     #create memory
#     memory = SequentialMemory(limit=25000, window_length=1)
#     #open files
#     # files = open("testFDQO.csv","w")
#     # files.write("kq\n")
#     #create callback
#     callbacks = CustomerTrainEpisodeLogger(folder + "callbacks_5phut.csv")
#     callback2 = ModelIntervalCheckpoint(folder + "weight_callbacks.h5f",interval=50000)
#     #callback3 = TestLogger11(files)
#     model.compile(Adam(learning_rate=1e-3), metrics=['mae'])
#     model.fit(env, nb_steps= 200000, visualize=False, verbose=2,callbacks=[callbacks,callback2],
#               baseline = baseline, eps = 1)
#     #model.fit(env, nb_steps= 130000, visualize=False, verbose=2,callbacks=[callbacks,callback2])
#     #files.close()

# if __name__=="__main__":
#     #file = "csvFilesNorm200steps" # Location to save all the results
#     # types = "DQL"
#     # if len(sys.argv) > 1:
#     #     types = sys.argv[1]
#     # if types =="FDQO":
#     #     Run_FDQO()
#     # elif types == "Random":
#     #     Run_Random()
#     # elif types == "Fuzzy":
#     #     Run_Fuzzy()
#     # elif types == "DQL":
#     #     Run_DQL()
#     # elif types == "DDQL":
#     #     Run_DDQL()
#     #create model FDQO
#     for i in range(1,2):
#         try:
#             #Run_DQL("M900_1000_" + str(i), file)
#             #Run_BDQL("M900_1000_200steps_gamma_0.9_static", file)
#             #Run_DDQL("M900_1000_200steps_2", file)
#             Run_FDQO("M900_1000_0.9_g0.5")
#             #Run_FDQO("M900_1000_0.9_baseline0.4_queue1k5", file)
#             #Run_RGreedy("M900_1000_200_tslots", file)
#             #Run_Sarsa("M900_1000", file)
#         except:
#             continue
   
#     # for i in range(1,6):
#     #     try:
#     #         Run_DDQL("M900_1000_" + str(i), file)
#     #     except:
#     #         continue   
#     # for i in range(1,6):
#     #     try:
#     #         Run_FDQO("M900_1000_0.9_baseline"+i, file)
#     #     except:
#     #         continue  

