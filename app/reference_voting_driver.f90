program reference_voting_driver
  use sklearn
  use sklearn_common, only: require,unique_labels,label_index
  implicit none
  type(voting_classifier) :: clf,clf_copy
  type(voting_regressor) :: reg,reg_copy
  type(voting_estimator), allocatable :: bases(:)
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  type(sgd_classifier) :: sgd
  type(kneighbors_classifier) :: knn
  type(decision_tree_classifier) :: tree
  type(multinomial_nb) :: mnb
  type(bernoulli_nb) :: bnb
  type(complement_nb) :: cnb
  type(perceptron) :: percep
  type(passive_aggressive_classifier) :: pa
  type(ridge_classifier) :: rc
  type(linear_svc) :: lsvc
  type(svc) :: ksvc
  type(ridge_classifier_cv) :: rcv
  type(dummy_classifier) :: dc
  type(linear_regression) :: ols
  type(ridge) :: rr
  type(sgd_regressor) :: sgdr
  type(kneighbors_regressor) :: knnr
  type(decision_tree_regressor) :: tr
  type(elastic_net) :: en
  type(lasso) :: la
  type(passive_aggressive_regressor) :: par
  type(kernel_ridge) :: kr
  type(dummy_regressor) :: dr
  real(dp), allocatable :: x(:,:),q(:,:),yr(:),truth(:),w(:),sw(:),vw(:),response(:,:),before(:,:)
  integer, allocatable :: y(:),choices(:),drops(:),classes(:),pre_y(:)
  integer :: input,output,n,p,m,b,mode,weighted,vweights,refit,seed,case_,i,j,stage
  character(1024) :: path,outpath
  call get_command_argument(1,path); call get_command_argument(2,outpath)
  open(newunit=input,file=trim(path),status='old')
  read(input,*) mode,n,p,m,b,weighted,vweights,refit,seed,case_
  allocate(x(n,p),q(m,p),yr(n),y(n),truth(m),w(n),sw(m),vw(b),choices(b),drops(b),bases(b))
  read(input,*) choices; read(input,*) drops; read(input,*) vw; read(input,*) w; read(input,*) sw
  do i=1,n
    read(input,*) x(i,:)
  end do
  read(input,*) yr; y=int(yr)
  do i=1,m
    read(input,*) q(i,:)
  end do
  read(input,*) truth; close(input)
  lr%c=.7_dp; lr%tol=1.e-10_dp; lr%max_iter=500
  sgd%loss='log_loss'; sgd%learning_rate='constant'; sgd%eta0=.02_dp; sgd%alpha=.001_dp
  sgd%max_iter=4; sgd%tol=-1; sgd%shuffle=.false.
  if(case_==3) then
    sgd%loss='modified_huber'; sgd%average=3
  end if
  knn%n_neighbors=3; knn%weights='distance'
  tree%max_depth=2; tree%min_samples_leaf=4
  mnb%alpha=.4_dp; bnb%alpha=.4_dp; cnb%alpha=.4_dp; cnb%norm=.true.
  percep%eta0=.1_dp; percep%max_iter=4
  pa%c=.7_dp; pa%max_iter=4; pa%tol=-1; pa%shuffle=.false.
  rc%alpha=.7_dp; rcv%alphas=[.1_dp,.7_dp,2._dp]
  lsvc%c=.7_dp; lsvc%tol=1.e-10_dp; lsvc%max_iter=10000
  ksvc%c=.7_dp; ksvc%gamma=.4_dp; ksvc%tol=1.e-8_dp; ksvc%max_iter=100000
  rr%alpha=.7_dp
  sgdr%learning_rate='constant'; sgdr%eta0=.02_dp; sgdr%alpha=.001_dp
  sgdr%max_iter=4; sgdr%tol=-1; sgdr%shuffle=.false.
  if(case_==3) sgdr%average=3
  knnr%n_neighbors=3; knnr%weights='distance'; tr%max_depth=2; tr%min_samples_leaf=4
  en%alpha=.05_dp; en%l1_ratio=.3_dp; en%tol=1.e-10_dp; en%max_iter=10000
  la%alpha=.05_dp; la%tol=1.e-10_dp; la%max_iter=10000
  par%c=.7_dp; par%max_iter=4; par%tol=-1; par%shuffle=.false.
  kr%kernel='rbf'; kr%gamma=.4_dp; kr%alpha=.7_dp
  if(case_==3) then
    dc%strategy='constant'; dc%constant=0; dr%strategy='constant'; dr%constant=2
  end if
  classes=unique_labels(y); pre_y=[(mod(label_index(classes,y(i)),size(classes)),i=1,n)]
  do j=1,b
    write(bases(j)%name,'(a,i2.2)') 'base_',j
    bases(j)%drop=drops(j)/=0
    if(bases(j)%drop) cycle
    if(mode<2) then
      select case(choices(j))
      case(0)
        allocate(bases(j)%estimator,source=lr)
      case(1)
        allocate(bases(j)%estimator,source=nb)
      case(2)
        allocate(bases(j)%estimator,source=sgd)
      case(3)
        allocate(bases(j)%estimator,source=knn)
      case(4)
        allocate(bases(j)%estimator,source=tree)
      case(5)
        allocate(bases(j)%estimator,source=mnb)
      case(6)
        allocate(bases(j)%estimator,source=bnb)
      case(7)
        allocate(bases(j)%estimator,source=cnb)
      case(8)
        allocate(bases(j)%estimator,source=percep)
      case(9)
        allocate(bases(j)%estimator,source=pa)
      case(10)
        allocate(bases(j)%estimator,source=rc)
      case(11)
        allocate(bases(j)%estimator,source=lsvc)
      case(12)
        allocate(bases(j)%estimator,source=ksvc)
      case(13)
        allocate(bases(j)%estimator,source=rcv)
      case(14)
        allocate(bases(j)%estimator,source=dc)
      end select
    else
      select case(choices(j))
      case(0)
        allocate(bases(j)%estimator,source=ols)
      case(1)
        allocate(bases(j)%estimator,source=rr)
      case(2)
        allocate(bases(j)%estimator,source=sgdr)
      case(3)
        allocate(bases(j)%estimator,source=knnr)
      case(4)
        allocate(bases(j)%estimator,source=tr)
      case(5)
        allocate(bases(j)%estimator,source=en)
      case(6)
        allocate(bases(j)%estimator,source=la)
      case(7)
        allocate(bases(j)%estimator,source=par)
      case(8)
        allocate(bases(j)%estimator,source=kr)
      case(9)
        allocate(bases(j)%estimator,source=dr)
      end select
    end if
    call fit_template(bases(j)%estimator)
  end do
  open(newunit=output,file=trim(outpath),status='replace')
  do j=1,b
    if(bases(j)%drop) cycle
    call model_state(bases(j)%estimator)
    call vector(predict_template(bases(j)%estimator))
  end do
  if(vweights/=0) then
    clf%weights=vw; reg%weights=vw
  end if
  if(mode==1) clf%voting='soft'
  do stage=0,refit
    if(stage==1) then
      if(mode<2) then
        y=merge(-31,83,y==classes(1))
      else
        yr=1.3_dp*yr+2
      end if
    end if
    if(mode<2) then
      if(weighted/=0) then
        call clf%fit(x,y,bases,w)
      else
        call clf%fit(x,y,bases)
      end if
      clf_copy=clf
      write(output,*) size(clf%classes_),size(clf%estimators_),clf%n_features_in_
      write(output,*) clf%classes_; write(output,*) clf%active_indices_
      call vector(real(clf%predict(q),dp)); call matrix(clf%transform(q))
      if(mode==1) then
        call matrix(clf%predict_proba(q))
        block
          real(dp), allocatable :: tensor(:,:,:)
          clf%flatten_transform=.false.; tensor=clf%transform_proba(q)
          do j=1,size(tensor,1)
            call matrix(tensor(j,:,:))
          end do
          clf%flatten_transform=.true.
        end block
      end if
      call vector([clf%score(q,int(truth)),clf%score(q,int(truth),sw)])
      do j=1,size(clf%estimators_)
        write(output,*) trim(clf%estimators_(j)%name)
        call model_state(clf%estimators_(j)%estimator)
      end do
      call vector(real(clf_copy%predict(q),dp))
    else
      if(weighted/=0) then
        call reg%fit(x,yr,bases,w)
      else
        call reg%fit(x,yr,bases)
      end if
      reg_copy=reg
      write(output,*) 0,size(reg%estimators_),reg%n_features_in_
      write(output,*) reg%active_indices_
      call vector(reg%predict(q)); call matrix(reg%transform(q))
      call vector([reg%score(q,truth),reg%score(q,truth,sw)])
      do j=1,size(reg%estimators_)
        write(output,*) trim(reg%estimators_(j)%name)
        call model_state(reg%estimators_(j)%estimator)
      end do
      call vector(reg_copy%predict(q))
    end if
  end do
  do j=1,b
    if(bases(j)%drop) cycle
    call model_state(bases(j)%estimator)
    call vector(predict_template(bases(j)%estimator))
  end do
  call matrix(x); call matrix(q); call vector(w); call vector(vw); call vector(sw)
  if(mode<2) then
    call vector(real(y,dp))
  else
    call vector(yr)
  end if
  close(output)
contains
  subroutine vector(a)
    real(dp), intent(in) :: a(:)
    write(output,'(*(es25.16e3,1x))') a
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: row
    do row=1,size(a,1)
      call vector(a(row,:))
    end do
  end subroutine
  subroutine fit_template(model)
    class(*), intent(inout) :: model
    select type(model)
    type is(logistic_regression)
      call model%fit(x,pre_y)
    type is(gaussian_nb)
      call model%fit(x,pre_y)
    type is(sgd_classifier)
      call model%fit(x,pre_y)
    type is(kneighbors_classifier)
      call model%fit(x,pre_y)
    type is(decision_tree_classifier)
      call model%fit(x,pre_y)
    class is(multinomial_nb)
      call model%fit(x,pre_y)
    type is(perceptron)
      call model%fit(x,pre_y)
    type is(passive_aggressive_classifier)
      call model%fit(x,pre_y)
    class is(ridge_classifier)
      call model%fit(x,pre_y)
    type is(linear_svc)
      call model%fit(x,pre_y)
    type is(svc)
      call model%fit(x,pre_y)
    type is(dummy_classifier)
      call model%fit(x,pre_y)
    class is(linear_regression)
      call model%fit(x,yr+3)
    type is(sgd_regressor)
      call model%fit(x,yr+3)
    type is(kneighbors_regressor)
      call model%fit(x,yr+3)
    type is(decision_tree_regressor)
      call model%fit(x,yr+3)
    type is(passive_aggressive_regressor)
      call model%fit(x,yr+3)
    type is(kernel_ridge)
      call model%fit(x,yr+3)
    type is(dummy_regressor)
      call model%fit(x,yr+3)
    end select
  end subroutine
  function predict_template(model) result(values)
    class(*), intent(in) :: model
    real(dp), allocatable :: values(:)
    select type(model)
    type is(logistic_regression)
      values=real(model%predict(q),dp)
    type is(gaussian_nb)
      values=real(model%predict(q),dp)
    type is(sgd_classifier)
      values=real(model%predict(q),dp)
    type is(kneighbors_classifier)
      values=real(model%predict(q),dp)
    type is(decision_tree_classifier)
      values=real(model%predict(q),dp)
    class is(multinomial_nb)
      values=real(model%predict(q),dp)
    type is(perceptron)
      values=real(model%predict(q),dp)
    type is(passive_aggressive_classifier)
      values=real(model%predict(q),dp)
    class is(ridge_classifier)
      values=real(model%predict(q),dp)
    type is(linear_svc)
      values=real(model%predict(q),dp)
    type is(svc)
      values=real(model%predict(q),dp)
    type is(dummy_classifier)
      values=real(model%predict(q),dp)
    class is(linear_regression)
      values=real(model%predict(q),dp)
    type is(sgd_regressor)
      values=real(model%predict(q),dp)
    type is(kneighbors_regressor)
      values=real(model%predict(q),dp)
    type is(decision_tree_regressor)
      values=real(model%predict(q),dp)
    type is(passive_aggressive_regressor)
      values=real(model%predict(q),dp)
    type is(kernel_ridge)
      values=real(model%predict(q),dp)
    type is(dummy_regressor)
      values=real(model%predict(q),dp)
    end select
  end function
  subroutine model_state(model)
    class(*), intent(in) :: model
    integer :: j
    select type(model)
    type is(logistic_regression)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector(model%intercept_)
    type is(gaussian_nb)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 2,size(model%theta_,1),size(model%theta_,2)
      call matrix(model%theta_); call matrix(model%var_)
      call vector(model%class_prior_); call vector(model%class_count_)
    type is(sgd_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 3,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector([model%intercept_,model%t_])
    type is(kneighbors_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 6,size(model%search_%x_,1),size(model%search_%x_,2)
      call matrix(model%search_%x_); call vector(real(model%y_,dp))
    type is(decision_tree_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 5,size(model%classes_),model%node_count_
      do j=1,model%node_count_
        associate(node=>model%nodes_(j))
          write(output,'(5(i0,1x),*(es25.16e3,1x))') &
            node%left,node%right,node%feature,node%n_samples,node%depth,node%threshold,node%impurity,node%probability
        end associate
      end do
    class is(multinomial_nb)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 4,size(model%feature_log_prob_,1),size(model%feature_log_prob_,2)
      call matrix(model%feature_log_prob_); call vector(model%class_log_prior_)
    type is(perceptron)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector(model%intercept_)
    type is(passive_aggressive_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 3,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector([model%intercept_,model%t_])
    class is(ridge_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector(model%intercept_)
    type is(linear_svc)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 1,size(model%coef_,1),size(model%coef_,2)
      call matrix(model%coef_); call vector(model%intercept_)
    type is(svc)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 8,size(model%x_fit_,1),size(model%x_fit_,2)
      call matrix(model%x_fit_); call vector(model%models_(1)%dual)
      call vector([model%models_(1)%intercept])
    class is(linear_regression)
      write(output,*) 0
      write(output,*) 9,1,size(model%coef_)
      call vector(model%coef_); call vector([model%intercept_])
    type is(sgd_regressor)
      write(output,*) 0
      write(output,*) 11,1,size(model%coef_)
      call vector(model%coef_); call vector([model%intercept_,model%t_])
    type is(kneighbors_regressor)
      write(output,*) 0
      write(output,*) 6,size(model%search_%x_,1),size(model%search_%x_,2)
      call matrix(model%search_%x_); call vector(real(model%y_,dp))
    type is(decision_tree_regressor)
      write(output,*) 0
      write(output,*) 7,0,model%node_count_
      do j=1,model%node_count_
        associate(node=>model%nodes_(j))
          write(output,'(5(i0,1x),3(es25.16e3,1x))') &
            node%left,node%right,node%feature,node%n_samples,node%depth,node%threshold,node%value,node%impurity
        end associate
      end do
    type is(dummy_classifier)
      write(output,*) size(model%classes_)
      write(output,*) model%classes_
      write(output,*) 12,0,0
      call vector(model%class_prior_)
    type is(dummy_regressor)
      write(output,*) 0
      write(output,*) 13,0,0
      call vector([model%constant_])
    type is(kernel_ridge)
      write(output,*) 0
      write(output,*) 14,size(model%x_fit_,1),size(model%x_fit_,2)
      call matrix(model%x_fit_); call vector(model%dual_coef_)
    type is(passive_aggressive_regressor)
      write(output,*) 0
      write(output,*) 11,1,size(model%coef_)
      call vector(model%coef_); call vector([model%intercept_,model%t_])
    end select
  end subroutine
end program
