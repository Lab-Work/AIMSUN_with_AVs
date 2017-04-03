# This script copies and organizes data from the estimation result folder to Video folder. 



import os
import shutil
import tempfile

# sce_seeds = [(75, 3252), (75, 59230), (100, 24654), (100, 45234)]
sce_seeds = [(0,2143)]

result_dir = '../Estimation/Result/Full_result_rv/'
video_dir = '../Video/'

prop = ['Density', 'W']
mdl = ['1st', '2nd']

for sce, seed in sce_seeds:

	if not os.path.exists(video_dir + 'Video_sce{0}_seed{1}_data/'.format(sce, seed)):
		os.mkdir(video_dir + 'Video_sce{0}_seed{1}_data/'.format(sce, seed))

	for run in range(0, 10):

		for p in prop:
			for m in mdl:
				src = result_dir + 'Estimation_corrected_{0}/Estimation{1}_PR_{2}_Seed{3}_{4}.npy'.format(run, p, sce, seed, m)

				target = video_dir + 'Video_sce{0}_seed{1}_data/'.format(sce, seed) + 'Estimation{0}_PR_{1}_Seed{2}_{3}_{4}.npy'.format(p, sce, seed, m, run)
		
				# copy files
				shutil.copy(src, target)

		if run == 0:
			# copy the true state
			src_true = result_dir + 'Estimation_corrected_{0}/TrueDensity_PR_{1}_Seed{2}_1st.npy'.format(run, sce, seed)
			target_true = video_dir + 'Video_sce{0}_seed{1}_data/'.format(sce, seed) + 'TrueDensity_PR_{0}_Seed{1}.npy'.format(sce, seed)
			shutil.copy(src_true, target_true)


	print('Finished sec {0} seed {1}'.format(sce, seed))
