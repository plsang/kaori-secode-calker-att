function eval_generate_detection_csv()
	feature_list = {'densetrajectory.mbh.cb256.fc.l2',
		'densetrajectory.mbh.cb4000.soft.l2',
		'fusion-multiseg.mbh_fc',
		'fusion-multiseg.mbh_soft',
		'fusion-multiseg.sift_fc',
		'fusion-multiseg.sift_soft',
		'fusion.mbh_fc.mbh_soft',
		'fusion.mbh_fc.mbh_soft.sift_fc.sift_soft.mfcc_fc.mfcc_soft',
		'fusion.mbh_fc.sift_fc.mfcc_fc',
		'fusion.mbh_soft.sift_soft.mfcc_soft',
		'fusion.mfcc_fc.mfcc_soft',
		'fusion.multiseg_mbh_fc.multiseg_mbh_soft.multiseg_sift_fc.multiseg_sift_soft.mfcc_fc.mfcc_soft',
		'fusion.multiseg_mbh_fc.multiseg_sift_fc.mfcc_fc',
		'fusion.multiseg_mbh_soft.multiseg_sift_soft.mfcc_soft',
		'fusion.sift_fc.sift_soft',
		'mfcc.rastamat.cb256.fc.l2',
		'mfcc.rastamat.cb4000.soft.l2',
		'fusion.mbh_fc.sift_fc',
		'fusion.multiseg_mbh_fc.multiseg_sift_fc',
		'fusion.mbh_fc.mbh_soft.sift_fc.sift_soft',
		'fusion.multiseg_mbh_fc.multiseg_mbh_soft.multiseg_sift_fc.multiseg_sift_soft'
		};
	
	for ii = 1:length(feature_list),
		feature_name = feature_list{ii};
		fprintf('Gen detection csv for feature [%s] ...\n', feature_name);
		eval_generate_detection_csv_(feature_name);
	end
			
end

function eval_generate_detection_csv_(feature_name)
	
	proj_dir = '/net/per610a/export/das11f/plsang';
	proj_name = 'trecvidmed13';
	exp_name = 'trecvidmed13-100000';
	suffix = '--calker-v7.1';
	
	% eval_generate_detection_csv('trecvidmed11', 'trecvidmed11-100000', 'densetrajectory.mbh.Soft-4000-VL2.MBH.trecvidmed11.devel.kcb', 'l2')
	
	addpath('/net/per900a/raid0/plsang/tools/kaori-secode-calker-v7.1/support');
	
	calker_exp_dir = sprintf('%s/%s/experiments/%s-calker/%s%s', proj_dir, proj_name, exp_name, feature_name, suffix);

	f_detection_csv = sprintf('%s/scores/test/%s.detection.csv', calker_exp_dir, feature_name);
	f_detection_csv
	
	if exist(f_detection_csv, 'file'),
		fprintf('File already exist! skipped!\n');
		return;
	end
	
	f_trial_index = '/net/per610a/export/das11f/plsang/dataset/MED2013/MEDDATA/databases/PROGTEST_20120507_TrialIndex.csv';
	
	fprintf('Reading trial index...\n');
	fh = fopen(f_trial_index);
	trial_indexes = textscan(fh, '%s %s %s', 'delimiter', ',');
	fclose(fh);
	
	trial_ids = trial_indexes{1}(2:end);
	video_num_ids = trial_indexes{2}(2:end);
	event_ids = trial_indexes{3}(2:end);
	
	
	video_idx_file = '/net/per610a/export/das11f/plsang/trecvidmed13/metadata/common/metadata_test_index.mat';
	
	if ~exist(video_idx_file, 'file'),
		test_lst = '/net/per610a/export/das11f/plsang/trecvidmed13/metadata/common/trecvidmed13.test.lst';
		
		fprintf('Generating video index...\n');
		test_videos = textread(test_lst, '%s');
		video_idxs = struct;
		for ii = 1:length(test_videos),
			video_id = test_videos{ii};
			video_idxs.(video_id) = ii;
		end	
		save(video_idx_file, 'video_idxs');
	else
		fprintf('Loading video index...\n');
		load(video_idx_file, 'video_idxs');
	end
	

	
	videoScorePath = sprintf('%s/scores/test/%s.scores.mat', calker_exp_dir, feature_name);
	
	calker_common_exp_dir = sprintf('%s/%s/experiments/%s-calker/common/%s', proj_dir, proj_name, exp_name, feature_name);
	
	test_db_file = 'database_test.mat';
	
	gt_file = fullfile(calker_common_exp_dir, test_db_file);
	
	if ~exist(gt_file, 'file'),
		warning('File not found! [%s] USING COMMON DIR GROUNDTRUTH!!!', gt_file);
		calker_common_exp_dir = sprintf('%s/%s/experiments/%s-calker/common', proj_dir, proj_name, exp_name);
		gt_file = fullfile(calker_common_exp_dir, test_db_file);
	end
	
	fprintf('Loading database [%s]...\n', test_db_file);
    database = load(gt_file, 'database');
	database = database.database;
	
	fprintf('Loading scores...\n');
	scores = load(videoScorePath);
	events  = fieldnames(scores);
	
	fprintf('Saving detection scores...\n');
	fh = fopen(f_detection_csv, 'w');
	fprintf(fh, '"TrialID","Score"\n');
	
	for jj=1:length(trial_ids),
		if ~mod(jj, 50000),
			fprintf('%.2f %% ...  ', 100*jj/length(trial_ids));
		end
		event_id = strrep(event_ids{jj}, '"', '');
		
		video_num_id = strrep(video_num_ids{jj}, '"', '');
		
		video_name = ['HVC', video_num_id];
		
		if isfield(video_idxs, video_name),
			vid_idx = video_idxs.(video_name);
			score = scores.(event_id)(vid_idx);
		else
			score = 0;
		end
						
		fprintf(fh, '%s,"%f"\n', trial_ids{jj}, score);
	end	
	fprintf('Done...\n');
	fclose(fh);

end