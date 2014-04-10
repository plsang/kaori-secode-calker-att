function calker_main_rank(proj_name, exp_id, feature_ext, suffix, test_pat, feat_dim, ker_type, cross, open_pool)

addpath('/net/per900a/raid0/plsang/tools/kaori-secode-calker-v6/support');
addpath('/net/per900a/raid0/plsang/tools/libsvm-3.17/matlab');
addpath('/net/per900a/raid0/plsang/tools/vlfeat-0.9.16/toolbox');

% run vl_setup with no prefix
% vl_setup('noprefix');
vl_setup;

%proj_name = 'trecvidmed11';
%exp_name = 'trecvidmed11-100000';
%kf_name = 'keyframe-100000';
exp_name = [proj_name, '-', exp_id];
seg_name = ['segment-', exp_id];

event_list = '/net/per610a/export/das11f/plsang/trecvidmed13/metadata/common/trecvidmed13.events.ps.lst';
fh = fopen(event_list, 'r');
infos = textscan(fh, '%s %s', 'delimiter', ' >.< ', 'MultipleDelimsAsOne', 1);
fclose(fh);
events = infos{1};

if ~exist('suffix', 'var'),
	suffix = '--calker-v7.1';
end

if ~exist('feat_dim', 'var'),
	feat_dim = 4000;
end

if ~exist('ker_type', 'var'),
	ker_type = 'echi2';
end

if ~exist('cross', 'var'),
	cross = 0;
end

if ~exist('open_pool', 'var'),
	open_pool = 5;
end

if ~exist('test_pat', 'var'),
	test_pat = 'kindredtest';
end

if isempty(strfind(suffix, '-v7')),
	error('**** Suffix does not contain v7 !!!!!\n');
end

ker = calker_build_kerdb(feature_ext, ker_type, feat_dim, cross, suffix);

ker.events = events;
ker.event_set = 'EK130';	% EK10, EK100
ker.dev_pat = 'dev';
ker.test_pat = test_pat;
%ker.test_pat = 'medtest';

calker_exp_dir = sprintf('%s/%s/experiments/%s-calker/%s%s', ker.proj_dir, proj_name, exp_name, ker.feat, ker.suffix);
ker.log_dir = fullfile(calker_exp_dir, 'log');

videolevel = strcmp('100000', exp_id);

%calker_create_database(proj_name, exp_name, seg_name, ker);
%calker_create_traindb(proj_name, exp_name, ker);

%open pool
%if matlabpool('size') == 0 && open_pool > 0, matlabpool(open_pool); end;
%calker_cal_train_kernel(proj_name, exp_name, ker);
%calker_train_kernel(proj_name, exp_name, ker, events);
%calker_cal_test_kernel(proj_name, exp_name, ker);
%calker_test_kernel(proj_name, exp_name, ker, events);
%calker_cal_map(proj_name, exp_name, ker, events, videolevel);
if ~isempty(strfind(ker.name, 'fusion')),
	ker.name = ker.feat_raw;
	ker.feat = ker.feat_raw;
end

ker.test_pat = 'kindredtest';
calker_cal_rank(proj_name, exp_name, ker, events);
ker.test_pat = 'medtest';
calker_cal_rank(proj_name, exp_name, ker, events);
ker.test_pat = 'test';
calker_cal_rank(proj_name, exp_name, ker, events);
%close pool
if matlabpool('size') > 0, matlabpool close; end;
