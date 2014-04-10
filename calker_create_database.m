function calker_create_database(proj_name, exp_name, seg_name, ker)

common_dir = sprintf('%s/%s/metadata/common', ker.proj_dir, proj_name);
calker_exp_dir = sprintf('%s/%s/experiments/%s-calker/%s%s', ker.proj_dir, proj_name, exp_name, ker.feat, ker.suffix);
calker_common_exp_dir = sprintf('%s/%s/experiments/%s-calker/common/%s', ker.proj_dir, proj_name, exp_name, ker.feat);
if ~exist(calker_common_exp_dir, 'file'),
	mkdir(calker_common_exp_dir);
end


pats = [{'dev'} {'kindredtest'} {'medtest'}];

% load event list
events = ker.events;
% create db for dev part

skip_exist = 0;
use_ps_video_only = 1; % only use pre-specified videos, discard ad-hoc videos

for pat_ = pats,
    pat = pat_{:};
    
    db_file = fullfile(calker_common_exp_dir, ['database_' pat '.mat']);
	if exist(db_file, 'file'),
		fprintf('File [%s] already exist! skip!\n', db_file);
		continue;
	end
	
	segment_pat = 'devel';
	if strcmp(pat, 'test'),
		segment_pat = 'test';
	end
	
	if strcmp(pat, 'dev') && use_ps_video_only == 1,
		video_file = fullfile(common_dir, [proj_name '.' pat '.ps.lst']);
	else
		video_file = fullfile(common_dir, [proj_name '.' pat '.lst']);
	end
    segment_dir = sprintf('%s/%s/metadata/%s/%s', ker.proj_dir, proj_name, seg_name, segment_pat);
    
    if skip_exist && exist(db_file,'file')~=0,
        fprintf('Skipped creating db [%s]!!\n', db_file);
    else
        label_files = {};
        for jj = 1:length(events),
            event = events{jj};
			if strcmp(pat, 'dev'),
				label_file = fullfile(common_dir, pat, ker.event_set, [event '.' ker.event_set '.lst']); 	% label 1
			else
				label_file = fullfile(common_dir, pat, [event '.' pat '.lst']); 	% label 1
			end
            
            label_files = [label_files; label_file];
        end
        
        fprintf('creating the database...');

        database = [];

        database.imnum = 0; % total number of segments
        database.cname = {}; % name of each video
        database.label = []; % label of each video
        database.path = {}; % contain the pathes for each image of each class
        database.nclass = 0; % number of videos
        database.video = []; % video index for each segment

        % load video list
        videos = textread(video_file, '%s');
	
        % load ground-truth
		if ~strcmp(pat, 'test'),
			for ii = 1:length(label_files),
				%video_list = read_file(label_files{ii});
				
				if strcmp(pat, 'dev'),
					[pos_videos, miss_videos] = load_ann_event_videos(label_files{ii});	
				else	% search test
					pos_videos = textread(label_files{ii}, '%s');
				end
				label_videos{ii} = pos_videos; % choose positive videos only
			end
		end
		
		fprintf('\n');
        for ii = 1:length(videos),
            video_name = videos{ii};

            database.nclass = database.nclass + 1;
            if mod(ii, 100) == 0,
				fprintf('1 --> %d --> %d\r', ii, length(videos));
			end
            database.cname{database.nclass} = video_name;

            segment_file = fullfile(segment_dir, [video_name '.lst']);
			if ~exist(segment_file, 'file'),
				msg = sprintf('Segment file not found [%s]', segment_file);
				disp(msg);
				log(ker, msg);
				continue;
			end
            segments = textread(segment_file, '%s');
                     
            c_num = length(segments);

            database.imnum = database.imnum + c_num;
            
			if ~strcmp(pat, 'test'),
			
				label = getlabel (video_name, label_videos);

				database.label = [database.label, ones(1, c_num)*label];
			end
			
            database.video = [database.video, ones(1, c_num)*database.nclass];

            %
            for jj = 1:c_num,               
				
				c_path = sprintf('%s/%s/feature/%s/%s/%s/%s.mat',...
						ker.proj_dir, proj_name, seg_name, ker.feat_raw, segment_pat, video_name);    
				
                database.path = [database.path, c_path];
            end;    
            
            %c_path = sprintf('/net/per900a/raid0/plsang/%s/feature/%s/%s/%s/%s.%s.tar.gz',...
            %        proj_name, kf_name, ker.feat, pat, video_name, ker.feat);                  
            %database.path = [database.path, c_path];

        end;
		fprintf('\n');
        disp('done!');
        
        save(db_file, 'database');
        clear database;
    end
end

end

function [pos_videos, miss_videos] = load_ann_event_videos(ann_file),
	fh = fopen(ann_file, 'r');
	infos = textscan(fh, '%s %s', 'delimiter', ' >.< ', 'MultipleDelimsAsOne', 1);
	fclose(fh);
	all_videos = infos{1};
	ann_infos = infos{2};
	pos_idx = find(ismember(ann_infos, 'positive'));
	miss_idx = find(ismember(ann_infos, 'miss'));
	pos_videos = all_videos(pos_idx);
	miss_videos = all_videos(miss_idx);
	
end

function label = getlabel(vid, videos)

	label = 0;
	for ii = 1:length(videos),
		%% BUG found when using strfind or findstr which will find the first inclusion string
		% if(size(cell2mat(strfind(videos{ii}, vid))) ~= 0),
		%	label = ii;
		% end
		
		if strmatch(vid, videos{ii}, 'exact'),
			label = ii;
		end
	end
	
	if label == 0,
		label = length(videos) + 1; % background label
	end
end

function log (ker, msg)
	fh = fopen(sprintf('%s/%s.log', ker.log_dir, mfilename), 'a+');
    msg = [msg, ' at ', datestr(now), '\n'];
	fprintf(fh, msg);
	fclose(fh);
end

