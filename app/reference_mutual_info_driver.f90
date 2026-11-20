program reference_mutual_info_driver
  use sklearn_mutual_info, only: mutual_info_classif,mutual_info_regression,mutual_info_score,r_regression
  use sklearn_common, only: dp
  implicit none
  real(dp), allocatable :: x(:,:),y(:),score(:)
  integer, allocatable :: labels(:),predicted(:),flags(:),indices(:)
  integer :: mode,n,p,k,seed,representation,center,force,input,output,i
  character(512) :: file
  call get_command_argument(1,file); open(newunit=input,file=trim(file),status='old')
  call get_command_argument(2,file); open(newunit=output,file=trim(file),status='replace')
  read(input,*) mode,n,p,k,seed,representation,center,force
  if(mode==4) then
    allocate(labels(n),predicted(n))
    if(n>0) then
      read(input,*) labels; read(input,*) predicted
    end if
    score=[mutual_info_score(labels,predicted)]
  else
    allocate(x(n,p),y(n),labels(n),flags(p))
    read(input,*) flags
    indices=pack([(i,i=1,p)],flags/=0)
    if(size(indices)>0) indices=[indices(size(indices):1:-1),indices(1)]
    do i=1,n
      read(input,*) x(i,:)
    end do
    if(mode==1) then
      read(input,*) labels
      select case(representation)
      case(0); score=mutual_info_classif(x,labels,n_neighbors=k,random_state=seed)
      case(1); score=mutual_info_classif(x,labels,discrete_features=flags/=0,n_neighbors=k,random_state=seed)
      case(2); score=mutual_info_classif(x,labels,discrete_features=all(flags/=0),n_neighbors=k,random_state=seed)
      case(3); score=mutual_info_classif(x,labels,discrete_features=indices,n_neighbors=k,random_state=seed)
      end select
    else
      read(input,*) y
      if(mode==3) then
        score=r_regression(x,y,center=center/=0,force_finite=force/=0)
      else
        select case(representation)
        case(0); score=mutual_info_regression(x,y,n_neighbors=k,random_state=seed)
        case(1); score=mutual_info_regression(x,y,discrete_features=flags/=0,n_neighbors=k,random_state=seed)
        case(2); score=mutual_info_regression(x,y,discrete_features=all(flags/=0),n_neighbors=k,random_state=seed)
        case(3); score=mutual_info_regression(x,y,discrete_features=indices,n_neighbors=k,random_state=seed)
        end select
      end if
    end if
  end if
  write(output,'(a,*(1x,es26.17e3))') 'scores',score
  if(mode/=4) write(output,'(a,*(1x,es26.17e3))') 'x',reshape(x,[size(x)])
  close(input); close(output)
end program
