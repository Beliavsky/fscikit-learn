program reference_multioutput_driver
  use sklearn
  implicit none
  type(multi_output_regressor) :: reg
  type(multi_output_classifier) :: clf
  type(linear_regression) :: ols
  type(ridge) :: ridge_base
  type(elastic_net) :: elastic
  type(sgd_regressor) :: sgdr
  type(kneighbors_regressor) :: knr
  type(decision_tree_regressor) :: tr
  type(logistic_regression) :: lr
  type(gaussian_nb) :: nb
  type(sgd_classifier) :: sgdc
  type(kneighbors_classifier) :: knc
  type(decision_tree_classifier) :: tc
  type(multinomial_nb) :: mn
  type(bernoulli_nb) :: bn
  type(complement_nb) :: cn
  type(label_set), allocatable :: classes(:)
  type(multioutput_probability), allocatable :: probabilities(:)
  class(*), allocatable :: estimator
  real(dp), allocatable :: x(:,:),query(:,:),ry(:,:),rq(:,:),weights(:),score_weights(:),before(:),after(:)
  real(dp), allocatable :: prediction(:,:)
  integer, allocatable :: cy(:,:),cq(:,:),cuts(:),labels(:,:)
  integer :: mode,base,n,p,m,t,batches,weighted,refit,shrink,probability,average,intercept,i,j,k,f,first,last,unit,out,top
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*) mode,base,n,p,m,t,batches,weighted,refit,shrink,probability,average,intercept
  allocate(x(n,p),query(m,p),weights(n),score_weights(m),cuts(batches),classes(t))
  read(unit,*) weights; read(unit,*) score_weights
  if(batches>0) read(unit,*) cuts
  do j=1,t
    read(unit,*) k
    allocate(classes(j)%values(k)); read(unit,*) classes(j)%values
  end do
  do i=1,n
    read(unit,*) x(i,:)
  end do
  do i=1,m
    read(unit,*) query(i,:)
  end do
  if(mode==0) then
    allocate(ry(n,t),rq(m,t))
    do i=1,n
      read(unit,*) ry(i,:)
    end do
    do i=1,m
      read(unit,*) rq(i,:)
    end do
  else
    allocate(cy(n,t),cq(m,t))
    do i=1,n
      read(unit,*) cy(i,:)
    end do
    do i=1,m
      read(unit,*) cq(i,:)
    end do
  end if
  close(unit)
  sgdr%learning_rate='constant'; sgdr%eta0=.02_dp; sgdr%alpha=.001_dp; sgdr%max_iter=4; sgdr%tol=-1
  sgdr%shuffle=.false.; sgdr%average=average; sgdr%fit_intercept=intercept==1
  sgdc%learning_rate='constant'; sgdc%eta0=.02_dp; sgdc%alpha=.001_dp; sgdc%max_iter=4; sgdc%tol=-1
  sgdc%shuffle=.false.; sgdc%average=average; sgdc%fit_intercept=intercept==1; sgdc%loss='log_loss'
  if(probability==0) sgdc%loss='hinge'
  ridge_base%alpha=.7_dp; elastic%alpha=.03_dp; elastic%l1_ratio=.4_dp; elastic%tol=1.e-12_dp
  elastic%max_iter=20000; knr%n_neighbors=3; knr%weights='distance'; knc%n_neighbors=3; knc%weights='distance'
  tr%max_depth=3; tc%max_depth=4; lr%c=.7_dp; lr%tol=1.e-10_dp; lr%max_iter=500
  mn%alpha=.4_dp; bn%alpha=.4_dp; cn%alpha=.4_dp; cn%norm=.true.
  if(mode==0) then
    select case(base)
    case(0)
      allocate(estimator,source=ols)
    case(1)
      allocate(estimator,source=ridge_base)
    case(2)
      allocate(estimator,source=elastic)
    case(3)
      allocate(estimator,source=sgdr)
    case(4)
      allocate(estimator,source=knr)
    case(5)
      allocate(estimator,source=tr)
    end select
    call prefit_reg(estimator); before=reg_base_predict(estimator)
  else
    select case(base)
    case(0)
      allocate(estimator,source=lr)
    case(1)
      allocate(estimator,source=nb)
    case(2)
      allocate(estimator,source=sgdc)
    case(3)
      allocate(estimator,source=knc)
    case(4)
      allocate(estimator,source=tc)
    case(5)
      allocate(estimator,source=mn)
    case(6)
      allocate(estimator,source=bn)
    case(7)
      allocate(estimator,source=cn)
    end select
    ! Pretrained templates must not leak fitted state into the initial clone.
    call clf%fit(x,cy(:,:1),estimator)
    deallocate(estimator); allocate(estimator,source=clf%estimators_(1)%estimator_)
    labels=clf%predict(query); before=real(labels(:,1),dp); clf=multi_output_classifier()
  end if
  open(newunit=out,file=trim(output),status='replace')
  if(batches==0) then
    call fit_all(); call snapshot()
  else
    first=1
    do f=1,batches
      last=cuts(f)
      if(mode==0) then
        if(weighted==1) then
          call reg%partial_fit(x(first:last,:),ry(first:last,:),estimator,sample_weight=weights(first:last))
        else
          call reg%partial_fit(x(first:last,:),ry(first:last,:),estimator)
        end if
      else
        if(weighted==1) then
          call clf%partial_fit(x(first:last,:),cy(first:last,:),estimator,classes,weights(first:last))
        else
          call clf%partial_fit(x(first:last,:),cy(first:last,:),estimator,classes)
        end if
      end if
      call snapshot(); first=last+1
    end do
    if(shrink==1.and.t>1) then
      if(mode==0) then
        call reg%partial_fit(x,ry(:,:t-1))
      else
        call clf%partial_fit(x,cy(:,:t-1))
      end if
      call snapshot()
    end if
    if(refit==1) then
      call fit_all(); call snapshot()
    end if
  end if
  if(mode==0) then
    after=reg_base_predict(estimator)
  else
    after=real(clf_base_predict(estimator),dp)
  end if
  write(out,'(*(es25.16e3,1x))') before; write(out,'(*(es25.16e3,1x))') after
  call matrix(x); call matrix(query); write(out,'(*(es25.16e3,1x))') weights
  if(mode==0) then
    call matrix(ry)
  else
    do i=1,n
      write(out,*) cy(i,:)
    end do
  end if
  close(out)
contains
  subroutine fit_all()
    if(mode==0) then
      if(weighted==1) then
        call reg%fit(x,ry,estimator,weights)
      else
        call reg%fit(x,ry,estimator)
      end if
    else
      if(weighted==1) then
        call clf%fit(x,cy,estimator,weights)
      else
        call clf%fit(x,cy,estimator)
      end if
    end if
  end subroutine
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: row
    do row=1,size(a,1)
      write(out,'(*(es25.16e3,1x))') a(row,:)
    end do
  end subroutine
  subroutine snapshot()
    integer :: output_count,c,r
    if(mode==0) then
      output_count=reg%n_outputs_; write(out,*) output_count,reg%n_features_in_,0
      prediction=reg%predict(query); call matrix(prediction)
      write(out,'(*(es25.16e3,1x))') reg%score(query,rq(:,:output_count)), &
        reg%score(query,rq(:,:output_count),score_weights)
      do c=1,output_count
        select type(model=>reg%estimators_(c)%estimator_)
        class is(linear_regression)
          write(out,*) 1
          write(out,'(*(es25.16e3,1x))') model%coef_,model%intercept_,0._dp
        type is(sgd_regressor)
          write(out,*) 1
          write(out,'(*(es25.16e3,1x))') model%coef_,model%intercept_,model%t_
        type is(decision_tree_regressor)
          write(out,*) 2,model%node_count_
          do r=1,model%node_count_
            associate(node=>model%nodes_(r))
              write(out,*) node%left,node%right,node%feature,node%n_samples,node%depth, &
                node%threshold,node%value,node%impurity
            end associate
          end do
        class default
          write(out,*) 0
        end select
      end do
    else
      output_count=clf%n_outputs_; top=0
      if(allocated(clf%classes_)) top=size(clf%classes_)
      write(out,*) output_count,clf%n_features_in_,top
      labels=clf%predict(query)
      do r=1,m
        write(out,*) labels(r,:)
      end do
      write(out,'(es25.16e3)') clf%score(query,cq(:,:output_count))
      if(probability==1) then
        probabilities=clf%predict_proba(query)
        do c=1,output_count
          write(out,*) size(probabilities(c)%classes); write(out,*) probabilities(c)%classes
          call matrix(probabilities(c)%values)
        end do
      end if
    end if
  end subroutine
  subroutine prefit_reg(model)
    class(*), intent(inout) :: model
    select type(model)
    class is(linear_regression)
      call model%fit(x,ry(:,1))
    type is(sgd_regressor)
      call model%fit(x,ry(:,1))
    type is(kneighbors_regressor)
      call model%fit(x,ry(:,1))
    type is(decision_tree_regressor)
      call model%fit(x,ry(:,1))
    end select
  end subroutine
  function clf_base_predict(model) result(values)
    class(*), intent(in) :: model
    integer, allocatable :: values(:)
    select type(model)
    type is(logistic_regression)
      values=model%predict(query)
    type is(gaussian_nb)
      values=model%predict(query)
    type is(sgd_classifier)
      values=model%predict(query)
    type is(kneighbors_classifier)
      values=model%predict(query)
    type is(decision_tree_classifier)
      values=model%predict(query)
    class is(multinomial_nb)
      values=model%predict(query)
    end select
  end function
  function reg_base_predict(model) result(values)
    class(*), intent(in) :: model
    real(dp), allocatable :: values(:)
    select type(model)
    class is(linear_regression)
      values=model%predict(query)
    type is(sgd_regressor)
      values=model%predict(query)
    type is(kneighbors_regressor)
      values=model%predict(query)
    type is(decision_tree_regressor)
      values=model%predict(query)
    end select
  end function
end program
