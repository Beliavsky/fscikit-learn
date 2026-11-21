program reference_logistic_cv_composites_driver
  use sklearn
  implicit none
  type(logistic_regression_cv) :: base
  type(gaussian_nb) :: nb
  type(voting_estimator) :: bases(2)
  type(voting_classifier) :: vote
  type(bagging_classifier) :: bag
  type(stacking_classifier) :: stack
  type(one_vs_rest_classifier) :: ovr
  type(multi_output_classifier) :: multi
  type(calibrated_classifier_cv) :: calibrated
  type(self_training_classifier) :: semi
  type(multioutput_probability), allocatable :: outputs(:)
  real(dp), allocatable :: x(:,:),query(:,:),w(:),prob(:,:),scores(:)
  integer, allocatable :: binary(:),many(:),targets(:,:),unlabeled(:),pred(:),predictions(:,:)
  integer :: input,output,n,p,m,i
  character(512) :: file
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) n,p,m
  allocate(x(n,p),query(m,p),w(n),binary(n),many(n),targets(n,2),unlabeled(n))
  do i=1,n
    read(input,*) x(i,:)
  end do
  read(input,*) binary; read(input,*) many; read(input,*) w
  do i=1,m
    read(input,*) query(i,:)
  end do
  targets(:,1)=binary; targets(:,2)=many; unlabeled=binary; unlabeled(1::4)=-1
  base%cs=[.1_dp,1._dp]; base%cv=2; base%tol=1.e-10_dp; base%max_iter=1000
  scores=cross_val_score(base,x,binary,cv=3); call emit('cross_val_score',scores)
  bases(1)%name='lrcv'; allocate(bases(1)%estimator,source=base)
  bases(2)%name='nb'; allocate(bases(2)%estimator,source=nb)
  vote%voting='soft'; call vote%fit(x,binary,bases,sample_weight=w)
  prob=vote%predict_proba(query); pred=vote%predict(query)
  call emit('voting_proba',reshape(prob,[size(prob)])); call emit('voting_predict',real(pred,dp))
  bag%n_estimators=2; bag%random_state=7; call bag%fit(x,binary,base,sample_weight=w)
  prob=bag%predict_proba(query); pred=bag%predict(query)
  call emit('bagging_proba',reshape(prob,[size(prob)])); call emit('bagging_predict',real(pred,dp))
  stack%cv=3; call stack%fit(x,binary,bases,final_estimator=base,sample_weight=w)
  prob=stack%predict_proba(query); pred=stack%predict(query)
  call emit('stacking_proba',reshape(prob,[size(prob)])); call emit('stacking_predict',real(pred,dp))
  call ovr%fit(x,many,base,sample_weight=w)
  prob=ovr%predict_proba(query); pred=ovr%predict(query)
  call emit('ovr_proba',reshape(prob,[size(prob)])); call emit('ovr_predict',real(pred,dp))
  call multi%fit(x,targets,base,sample_weight=w); outputs=multi%predict_proba(query); predictions=multi%predict(query)
  call emit('multi_predict',real(reshape(predictions,[size(predictions)]),dp))
  call emit('multi_proba1',reshape(outputs(1)%values,[size(outputs(1)%values)]))
  call emit('multi_proba2',reshape(outputs(2)%values,[size(outputs(2)%values)]))
  calibrated%cv=2; call calibrated%fit(x,binary,base,sample_weight=w)
  prob=calibrated%predict_proba(query); pred=calibrated%predict(query)
  call emit('calibrated_proba',reshape(prob,[size(prob)])); call emit('calibrated_predict',real(pred,dp))
  semi%max_iter=1; semi%criterion='k_best'; semi%k_best=4
  call semi%fit(x,unlabeled,base)
  prob=semi%predict_proba(query); pred=semi%predict(query)
  call emit('self_proba',reshape(prob,[size(prob)])); call emit('self_predict',real(pred,dp))
  call emit('transduction',real(semi%transduction_,dp)); call emit('labeled_iter',real(semi%labeled_iter_,dp))
contains
  subroutine emit(name,values)
    character(*), intent(in) :: name
    real(dp), intent(in) :: values(:)
    write(output,'(a,*(1x,es25.16e3))') name,values
  end subroutine
end program
