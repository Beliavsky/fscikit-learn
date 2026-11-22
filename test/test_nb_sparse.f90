program test_nb_sparse
  use sklearn_naive_bayes, only: multinomial_nb,bernoulli_nb,complement_nb
  use sklearn_sparse, only: csr_matrix,csr_from_arrays,csr_from_dense
  use sklearn_common, only: dp,require
  use, intrinsic :: ieee_arithmetic, only: ieee_is_nan,ieee_is_finite
  implicit none
  type(multinomial_nb) :: sparse,dense,stream
  type(bernoulli_nb) :: bern
  type(complement_nb) :: comp
  type(csr_matrix) :: x,q,zero,stored_zero,duplicates,large
  real(dp) :: values(4,3),query(2,3)
  real(dp), allocatable :: probability(:,:),joint(:,:)
  integer, allocatable :: labels(:)
  values(1,:)=[2._dp,0._dp,1._dp];values(2,:)=[1._dp,0._dp,2._dp]
  values(3,:)=[0._dp,3._dp,0._dp];values(4,:)=[0._dp,2._dp,1._dp]
  query(1,:)=[1._dp,0._dp,1._dp];query(2,:)=[0._dp,1._dp,1._dp]
  x=csr_from_dense(values);q=csr_from_dense(query)
  call sparse%fit(x,[-3,-3,9,9],sample_weight=[1._dp,2._dp,.5_dp,1._dp])
  call dense%fit(values,[-3,-3,9,9],sample_weight=[1._dp,2._dp,.5_dp,1._dp])
  call require(maxval(abs(sparse%predict_proba(q)-dense%predict_proba(query)))<1.e-13_dp,'CSR and dense positive smoothing')
  call stream%partial_fit(csr_from_dense(values(:2,:)),[-3,-3],classes=[9,-3],sample_weight=[1._dp,2._dp])
  call stream%partial_fit(values(3:,:),[9,9],sample_weight=[.5_dp,1._dp])
  call require(maxval(abs(stream%feature_count_-sparse%feature_count_))<1.e-14_dp,'CSR-to-dense streaming')
  stream=multinomial_nb();call stream%partial_fit(values(:2,:),[-3,-3],classes=[-3,9])
  call stream%partial_fit(csr_from_dense(values(3:,:)),[9,9])
  call dense%fit(values,[-3,-3,9,9])
  call require(maxval(abs(stream%predict_proba(q)-dense%predict_proba(query)))<1.e-13_dp,'dense-to-CSR streaming')
  duplicates=csr_from_arrays(4,3,[1,4,5,7,8],[3,1,1,3,2,2,3],[1._dp,1._dp,1._dp,2._dp,1._dp,2._dp,1._dp])
  call sparse%fit(duplicates,[-3,-3,9,9])
  call require(all(sparse%feature_count_(1,:)==[2._dp,0._dp,3._dp]),'unsorted duplicate count accumulation')
  call bern%fit(duplicates,[-3,-3,9,9])
  call require(bern%feature_count_(1,1)==2,'Bernoulli binarizes duplicate entries before summing')
  sparse%alpha=0;call sparse%fit(x,[-3,-3,9,9])
  zero=csr_from_arrays(1,3,[1,1],[integer::],[real(dp)::])
  stored_zero=csr_from_arrays(1,3,[1,2],[2],[0._dp])
  probability=sparse%predict_proba(zero)
  call require(all(ieee_is_finite(probability)),'implicit zero row skips infinite feature coefficients')
  probability=sparse%predict_proba(stored_zero)
  call require(all(ieee_is_nan(probability)),'stored zero times infinite coefficient remains source NaN')
  bern=bernoulli_nb();bern%alpha=0
  call bern%fit(csr_from_arrays(4,1,[1,2,3,3,3],[1,1],[1._dp,1._dp]),[-3,-3,9,9])
  zero=csr_from_arrays(1,1,[1,1],[integer::],[real(dp)::])
  probability=bern%predict_proba(zero)
  call require(probability(1,1)==0.and.probability(1,2)==1,'absent Bernoulli unit-probability feature')
  stored_zero=csr_from_arrays(1,1,[1,2],[1],[0._dp]);probability=bern%predict_proba(stored_zero)
  call require(probability(1,2)==1,'Bernoulli eliminates thresholded explicit zeros')
  comp%alpha=0;comp%norm=.true.;call comp%fit(x,[-3,-3,9,9])
  zero=csr_from_arrays(1,3,[1,1],[integer::],[real(dp)::]);joint=comp%joint_log_likelihood(zero)
  call require(all(joint==0),'empty Complement query skips NaN feature coefficients')
  large%n_rows=10000;large%n_cols=10000
  allocate(large%indptr(10001),large%indices(2),large%data(2),labels(10000))
  large%indptr=2;large%indptr(1)=1;large%indptr(5002:)=3
  large%indices=[1,10000];large%data=[2._dp,3._dp];labels(:5000)=-3;labels(5001:)=9
  sparse=multinomial_nb();call sparse%fit(large,labels)
  call require(all(sparse%class_count_==5000).and.sum(sparse%feature_count_)==5,'large sparse fit without dense samples')
  call require(sparse%n_features_in_==10000,'wide sparse fitted metadata')
  print *, 'CSR discrete Naive Bayes: PASS'
end program
