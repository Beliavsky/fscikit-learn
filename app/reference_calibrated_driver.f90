program reference_calibrated_driver
  use sklearn, only: dp,calibrated_classifier_cv,gaussian_nb,kneighbors_classifier,logistic_regression
  implicit none
  type(calibrated_classifier_cv) :: model
  type(gaussian_nb) :: nb
  type(kneighbors_classifier) :: knn
  type(logistic_regression) :: lr
  class(*), allocatable :: estimator,work
  real(dp), allocatable :: x(:,:),query(:,:),train_x(:,:),weights(:),prob(:,:),scores(:,:),raw(:,:),before(:,:),after(:,:)
  integer, allocatable :: y(:),train_y(:),ids(:),train(:),test(:)
  integer :: n,p,m,nt,base,method,ensemble,prefit,weighted,weighting,i,j,f,q,unit,out,nfit,rows
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*) n,p,m,nt,base,method,ensemble,prefit,model%cv,weighted,weighting
  allocate(x(n,p),query(m,p),train_x(nt,p),weights(n),y(n),train_y(nt))
  read(unit,*) y; read(unit,*) train_y; read(unit,*) weights
  do i=1,n
    read(unit,*) x(i,:)
  end do
  do i=1,nt
    read(unit,*) train_x(i,:)
  end do
  do i=1,m
    read(unit,*) query(i,:)
  end do
  close(unit)
  if(method==1) model%method='isotonic'
  model%ensemble=ensemble==1; model%prefit=prefit==1
  lr%c=.7_dp; lr%tol=1.e-10_dp; lr%max_iter=500
  knn%n_neighbors=3; if(weighting==1) knn%weights='distance'
  select case(base)
  case(0)
    call nb%fit(train_x,train_y); allocate(estimator,source=nb)
  case(1)
    call knn%fit(train_x,train_y); allocate(estimator,source=knn)
  case(2)
    call lr%fit(train_x,train_y); allocate(estimator,source=lr)
  end select
  before=base_probability(estimator,query)
  if(weighted==1) then
    call model%fit(x,y,estimator,weights)
  else
    call model%fit(x,y,estimator)
  end if
  after=base_probability(estimator,query); prob=model%predict_proba(query)
  q=size(model%classes_); if(q==2) q=1
  ids=[(i,i=1,n)]; allocate(scores(n,q))
  if(.not.model%ensemble.and..not.model%prefit) then
    do f=1,model%cv
      train=pack(ids,model%fold_id_/=f); test=pack(ids,model%fold_id_==f)
      allocate(work,source=estimator); call fit_base(work,x(train,:),y(train))
      raw=base_response(work,x(test,:)); scores(test,:)=raw; deallocate(work)
    end do
  end if
  open(newunit=out,file=trim(output),status='replace')
  write(out,*) size(model%calibrated_classifiers_),size(model%classes_)
  write(out,*) model%classes_
  write(out,*) model%predict(query)
  call matrix(prob)
  if(allocated(model%fold_id_)) then
    write(out,*) model%fold_id_
  else
    write(out,*) 0*ids
  end if
  do f=1,size(model%calibrated_classifiers_)
    if(model%prefit) then
      scores=base_response(model%calibrated_classifiers_(f)%estimator_,x)
    else if(model%ensemble) then
      test=pack(ids,model%fold_id_==f)
      scores=base_response(model%calibrated_classifiers_(f)%estimator_,x(test,:))
    end if
    nfit=size(scores,1); write(out,*) nfit,q; call matrix(scores)
    raw=base_response(model%calibrated_classifiers_(f)%estimator_,query); call matrix(raw)
    do j=1,q
      associate(map=>model%calibrated_classifiers_(f)%calibrators_(j))
        if(method==0) then
          write(out,'(*(es25.16e3,1x))') map%a_,map%b_
        else
          rows=size(map%isotonic_model_%x_thresholds_); write(out,*) rows
          write(out,'(*(es25.16e3,1x))') map%isotonic_model_%x_thresholds_
          write(out,'(*(es25.16e3,1x))') map%isotonic_model_%y_thresholds_
        end if
      end associate
    end do
  end do
  call matrix(before); call matrix(after)
  write(out,*) y; call matrix(x)
  close(out)
contains
  subroutine matrix(a)
    real(dp), intent(in) :: a(:,:)
    integer :: row
    do row=1,size(a,1)
      write(out,'(*(es25.16e3,1x))') a(row,:)
    end do
  end subroutine
  subroutine fit_base(a,b,c)
    class(*), intent(inout) :: a
    real(dp), intent(in) :: b(:,:)
    integer, intent(in) :: c(:)
    select type(a)
    type is(gaussian_nb)
      call a%fit(b,c)
    type is(kneighbors_classifier)
      call a%fit(b,c)
    type is(logistic_regression)
      call a%fit(b,c)
    end select
  end subroutine
  function base_probability(a,b) result(v)
    class(*), intent(in) :: a
    real(dp), intent(in) :: b(:,:)
    real(dp), allocatable :: v(:,:)
    select type(a)
    type is(gaussian_nb)
      v=a%predict_proba(b)
    type is(kneighbors_classifier)
      v=a%predict_proba(b)
    type is(logistic_regression)
      v=a%predict_proba(b)
    end select
  end function
  function base_response(a,b) result(v)
    class(*), intent(in) :: a
    real(dp), intent(in) :: b(:,:)
    real(dp), allocatable :: v(:,:),proba(:,:)
    select type(a)
    type is(logistic_regression)
      v=a%decision_function(b)
    class default
      proba=base_probability(a,b)
      if(size(proba,2)==2) then
        v=proba(:,2:2)
      else
        v=proba
      end if
    end select
  end function
end program
