warning off;
FILE_PATH = "C:\Users\rmappin\PhD_Data\Renal Project Images\_Batch1_Anon\";
subject_name = "UCLH_08933783";
study_date = "CT-20171021";
FILE_PATH = strcat(FILE_PATH, subject_name, "\", study_date, "\");

SAVE_PATH = "C:\Users\rmappin\PhD_Data\Virtual_Contrast_Data\Raw_NII_Vols\";
SAVE_PATH = strcat(SAVE_PATH, subject_name, "\");

batch_num = 1;
study_num = 1;
art_num = 3;
ven_num = 35;
non_num = 2;

ACE_save_name = strcat(subject_name, sprintf("_%d_%d_%d_ACE.nii", batch_num, study_num, art_num));
VCE_save_name = strcat(subject_name, sprintf("_%d_%d_%d_VCE.nii", batch_num, study_num, ven_num));
NCE_save_name = strcat(subject_name, sprintf("_%d_%d_%d_NCE.nii", batch_num, study_num, non_num));


%%
file_list = dir(FILE_PATH);
file_list = file_list(3:end);

cd(FILE_PATH);

CE_num = 0;
NCE_num = 0;
CE_files = cell(1, 250);
NCE_files = cell(1, 250);

for i = 1:length(file_list)
    meta = dicominfo(file_list(i).name);
    
    if meta.SeriesNumber == art_num
        CE_num = CE_num + 1;
        CE_files(CE_num) = {file_list(i).name};
    end
    
    if meta.SeriesNumber == non_num
        NCE_num = NCE_num + 1;
        NCE_files(NCE_num) = {file_list(i).name};
    end
end

CE_files = CE_files(~cellfun('isempty', CE_files));
NCE_files = NCE_files(~cellfun('isempty', NCE_files));

%%
CE_slice = zeros(1, length(CE_files));
NCE_slice = zeros(1, length(NCE_files));

for i = 1:length(CE_files)
    meta = dicominfo(CE_files{i});
    CE_slice(i) = meta.SliceLocation;
end

for i = 1:length(NCE_files)
    meta = dicominfo(NCE_files{i});
    NCE_slice(i) = meta.SliceLocation;
end

CE_slice(1:5)
NCE_slice(1:5)

%% If coord shift is needed
% NCE_slice = NCE_slice + 0.5;

%%
CE_min = min(CE_slice);
CE_max = max(CE_slice);
NCE_min = min(NCE_slice);
NCE_max = max(NCE_slice);

upper_bound = min([CE_max NCE_max]);
lower_bound = max([CE_min NCE_min]);

num_slice = 0;

for i = 1:length(CE_slice)
    idx = find(NCE_slice == CE_slice(i));
    if idx
        [CE_slice(i), NCE_slice(idx)];
        num_slice = num_slice + 1;
    end
end

coords = linspace(upper_bound, lower_bound, num_slice);

[CE_sorted, CE_idx] = sort(CE_slice, 'descend');
[NCE_sorted, NCE_idx] = sort(NCE_slice, 'descend');

CE_files_sorted = CE_files(CE_idx);
NCE_files_sorted = NCE_files(NCE_idx);

%%
CE_vol = zeros(512, 512, num_slice);
NCE_vol = zeros(512, 512, num_slice);

idx = 1;
% temp = CE_sorted(62:end);
% temp_files = CE_files_sorted(62:end);

for i = 1:length(CE_sorted)
    NCE_file_idx = find(NCE_sorted == CE_sorted(i));

    if NCE_file_idx
        CE_vol(:, :, idx) = double(dicomread(CE_files_sorted{i}));
        NCE_vol(:, :, idx) = double(dicomread(NCE_files_sorted{NCE_file_idx}));
        idx = idx + 1;
    end
end

%%
i = 50;
j = 250;
figure;
subplot(2, 3, 1), imshow(CE_vol(:, :, i), []);
subplot(2, 3, 1), title('Contrast enhanced');
subplot(2, 3, 2), imshow(NCE_vol(:, :, i), []);
subplot(2, 3, 2), title('Washout');
subplot(2, 3, 3), imshow(CE_vol(:, :, i) - NCE_vol(:, :, i), []);
subplot(2, 3, 3), title('Difference');
subplot(2, 3, 4), imshow(squeeze(CE_vol(j, :, :))', []);
subplot(2, 3, 4), title('Contrast enhanced');
subplot(2, 3, 5), imshow(squeeze(NCE_vol(j, :, :))', []);
subplot(2, 3, 5), title('Washout');
subplot(2, 3, 6), imshow(squeeze(CE_vol(j, :, :) - NCE_vol(j, :, :))', []);
subplot(2, 3, 6), title('Difference');

%%
figure;
pause(1);
for i = 1:num_slice
    subplot(1, 3, 1), imshow(CE_vol(:, :, i), []);
    subplot(1, 3, 2), imshow(NCE_vol(:, :, i), []);
    subplot(1, 3, 3), imshow(CE_vol(:, :, i) - NCE_vol(:, :, i), []);
    pause(0.05);
end

%%
cd(SAVE_PATH);
niftiwrite(CE_vol, ACE_save_name);
niftiwrite(NCE_vol, NCE_save_name);