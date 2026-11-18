program test_preprocessing_metrics
  use sklearn_common
  use sklearn_preprocessing
  use sklearn_metrics
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_quiet_nan
  implicit none
  type(standard_scaler) :: ss
  type(minmax_scaler) :: mm
  type(robust_scaler) :: rs
  type(maxabs_scaler) :: ma
  type(simple_imputer) :: im
  type(one_hot_encoder) :: oh
  type(label_encoder) :: le
  real(dp) :: x(4,2),z(4,2),y(4),p(4),missing(3,1)
  real(dp), allocatable :: poly(:,:),encoded(:,:)
  integer :: labels(4),pred(4)
  x(:,1)=[1._dp,2._dp,3._dp,4._dp]; x(:,2)=7
  call ss%fit(x); z=ss%transform(x)
  call require(maxval(abs(sum(z,dim=1)))<1.e-12_dp,'standard means')
  call require(abs(sum(z(:,1)**2)/4-1)<1.e-12_dp,'standard population variance')
  call require(maxval(abs(ss%inverse_transform(z)-x))<1.e-12_dp,'standard roundtrip')
  call mm%fit(x); z=mm%transform(x)
  call require(abs(z(4,1)-1)<1.e-12_dp.and.all(z(:,2)==0),'minmax constants')
  call require(maxval(abs(mm%inverse_transform(z)-x))<1.e-12_dp,'minmax roundtrip')
  call rs%fit(x); call require(abs(rs%center_(1)-2.5_dp)<1.e-12_dp,'robust median')
  call require(abs(rs%scale_(1)-1.5_dp)<1.e-12_dp,'robust IQR')
  call ma%fit(x); z=ma%transform(x)
  call require(maxval(abs(ma%inverse_transform(z)-x))<1.e-12_dp,'maxabs roundtrip')
  missing(:,1)=[1._dp,ieee_value(0._dp,ieee_quiet_nan),3._dp]
  call im%fit(missing); call require(all(abs(im%transform(missing)-2)<=1),'imputation')
  poly=polynomial_features(x,2)
  call require(size(poly,2)==6,'polynomial shape')
  call require(all(poly(:,5)==x(:,1)*x(:,2)),'polynomial feature order')
  poly=polynomial_features(x,3,interaction_only=.true.)
  call require(size(poly,2)==4,'interaction features')
  call oh%fit(x); encoded=oh%transform(x)
  call require(size(encoded,2)==5.and.all(sum(encoded,dim=2)==2),'one hot')
  call le%fit([9,-3,9]); call require(all(le%transform([9,-3])==[1,0]),'label encoding')
  call require(all(le%inverse_transform([0,1])==[-3,9]),'label decoding')
  y=[3._dp,-0.5_dp,2._dp,7._dp]; p=[2.5_dp,0._dp,2._dp,8._dp]
  call require(abs(mean_squared_error(y,p)-0.375_dp)<1.e-12_dp,'MSE')
  call require(abs(r2_score(y,p)-0.9486081370449679_dp)<1.e-12_dp,'R2')
  labels=[0,0,1,1]; pred=[0,1,1,1]
  call require(abs(accuracy_score(labels,pred)-0.75_dp)<1.e-12_dp,'accuracy')
  call require(abs(f1_score(labels,pred)-0.8_dp)<1.e-12_dp,'F1')
  call require(abs(precision_score(labels,pred)-2._dp/3)<1.e-12_dp,'precision')
  call require(abs(recall_score(labels,pred)-1)<1.e-12_dp,'recall')
  call require(abs(roc_auc_score(labels,[0.1_dp,0.4_dp,0.35_dp,0.8_dp])-0.75_dp)<1.e-12_dp,'AUC')
  call require(abs(average_precision_score(labels,[0.1_dp,0.4_dp,0.35_dp,0.8_dp])-5._dp/6)<1.e-12_dp,'AP')
  call require(abs(adjusted_rand_score(labels,[2,2,3,3])-1)<1.e-12_dp,'ARI label invariance')
  call require(abs(v_measure_score(labels,[2,2,3,3])-1)<1.e-12_dp,'V measure')
  call require(abs(silhouette_score(x,labels)-0.4666666666666667_dp)<1.e-12_dp,'silhouette')
  print *, 'preprocessing and metrics: PASS'
end program
