program reference_self_training_driver
  use sklearn, only: dp,self_training_classifier,gaussian_nb,kneighbors_classifier,logistic_regression
  implicit none
  type(self_training_classifier) :: model
  type(gaussian_nb) :: nb
  type(kneighbors_classifier) :: knn
  type(logistic_regression) :: lr
  real(dp), allocatable :: x(:,:),query(:,:),prob(:,:),logs(:,:),decision(:,:)
  integer, allocatable :: y(:),prediction(:)
  integer :: n,p,m,base,criterion,neighbors,weights,i,unit,out
  character(1024) :: input,output
  call get_command_argument(1,input); call get_command_argument(2,output)
  open(newunit=unit,file=trim(input),status='old')
  read(unit,*) n,p,m,base,criterion,model%max_iter,model%k_best,neighbors,weights,model%threshold
  allocate(x(n,p),query(m,p),y(n)); read(unit,*) y
  do i=1,n
    read(unit,*) x(i,:)
  end do
  do i=1,m
    read(unit,*) query(i,:)
  end do
  close(unit)
  if(criterion==1) model%criterion='k_best'
  knn%n_neighbors=neighbors; if(weights==1) knn%weights='distance'
  lr%c=.8_dp; lr%tol=1.e-10_dp; lr%max_iter=500
  select case(base)
  case(0)
    call model%fit(x,y,nb)
  case(1)
    call model%fit(x,y,knn)
  case(2)
    call model%fit(x,y,lr)
  end select
  prob=model%predict_proba(query); prediction=model%predict(query)
  if(base/=1) logs=model%predict_log_proba(query)
  open(newunit=out,file=trim(output),status='replace')
  write(out,*) model%n_iter_
  write(out,'(a)') trim(model%termination_condition_)
  write(out,*) size(model%classes_)
  write(out,*) model%classes_
  write(out,*) model%transduction_
  write(out,*) model%labeled_iter_
  write(out,*) prediction
  do i=1,m
    write(out,'(*(es25.16e3,1x))') prob(i,:)
  end do
  if(base/=1) then
    do i=1,m
      write(out,'(*(es25.16e3,1x))') logs(i,:)
    end do
  end if
  if(base==2) then
    decision=model%decision_function(query)
    do i=1,m
      write(out,'(*(es25.16e3,1x))') decision(i,:)
    end do
  end if
  write(out,*) y
  do i=1,n
    write(out,'(*(es25.16e3,1x))') x(i,:)
  end do
  close(out)
end program
