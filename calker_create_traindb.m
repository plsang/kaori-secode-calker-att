function calker_create_traindb(proj_name, exp_name, ker)

%proj_name = 'trecvidmed10';
%exp_name = 'keyframe-60';


meta_dir = sprintf('%s/%s/metadata', ker.proj_dir, proj_name);

calker_exp_dir = sprintf('%s/%s/experiments/%s-calker/%s%s', ker.proj_dir, proj_name, exp_name, ker.feat, ker.suffix);
calker_common_exp_dir = sprintf('%s/%s/experiments/%s-calker/common/%s', ker.proj_dir, proj_name, exp_name, ker.feat);

traindb_file = fullfile(calker_exp_dir, 'metadata', 'traindb.mat');
if exist(traindb_file, 'file')~= 0,
	fprintf('Traindb already exist!!\n');
    return;
end


dev_db_file = sprintf('database_%s.mat', ker.dev_pat);

db_file = fullfile(calker_common_exp_dir, dev_db_file);

fprintf('Loading database [%s]...\n', db_file);
load(db_file, 'database');

max_kf_per_non_bg_video = +inf; % only applied for background videos
max_kf_per_bg_video = +inf; % only applied for background videos
max_num_kf = +inf;
use_bg_video = 1;

ntype = unique(database.label);

% make sure non background clips appear first
num_non_bg_video = 0;
vids = [];
lbs = [];

for ii = ntype %
    vid_ii = unique(database.video(database.label == ii));
    vids = [vids, vid_ii];
    lbs = [lbs, ones(1, length(vid_ii))*ii];
    num_non_bg_video = num_non_bg_video + length(vid_ii);
end

num_non_bg_video = num_non_bg_video - length(vid_ii);

traindb.video = {}; 
traindb.sel = [];
traindb.label = [];
traindb.path = {};

for ii = 1:length(vids), %

	if ~use_bg_video && ii > num_non_bg_video, break; end;
	
    video = database.cname{vids(ii)};
    
	kfs_idx = find(database.video == vids(ii));
    n_kfs = length(kfs_idx);
    
	if ii <= num_non_bg_video && n_kfs > max_kf_per_non_bg_video		
		sel = randperm(n_kfs);
		sel = sel(1:max_kf_per_non_bg_video);
		n_kfs = max_kf_per_non_bg_video;
		
	elseif ii > num_non_bg_video && n_kfs > max_kf_per_bg_video
		sel = randperm(n_kfs);
		sel = sel(1:max_kf_per_bg_video);
		% or just simply: sel = randi(100, 1, max_kf_per_bg_video);
		n_kfs = max_kf_per_bg_video;
		
	else
		sel = [1:n_kfs];
	end
    
	
    traindb.video{ii} = video;
	traindb.sel.(video) = sel;
	traindb.label = [traindb.label, ones(1, n_kfs) * lbs(ii)];
	traindb.path = [traindb.path, database.path(kfs_idx(sel))];
	
	if length(traindb.label) > max_num_kf,
		break;
	end
end

% save seldb for next kernel computing
save(traindb_file, 'traindb');


end