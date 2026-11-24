program reference_class_weight_driver
  use sklearn_class_weight, only: compute_class_weight,compute_sample_weight,class_weight_map
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  use sklearn_common, only: dp,require
  implicit none
  integer :: cases,c,op,n,q,k,mode,m,ni,j,i,nkeys,pos
  integer, allocatable :: y(:,:),classes(:),indices(:),pointers(:),columns(:)
  real(dp), allocatable :: w(:),values(:)
  type(class_weight_map), allocatable :: maps(:),single
  type(csr_matrix) :: sparse
  character(8) :: choice
  read(*,*) cases
  do c=1,cases
    read(*,*) op,n,q,k,mode,m,ni
    allocate(y(n,q),classes(k));read(*,*) classes;read(*,*) y
    if(m>=0) then
      allocate(maps(m))
      do j=1,m
        read(*,*) nkeys;allocate(maps(j)%labels(nkeys),maps(j)%weights(nkeys))
        read(*,*) maps(j)%labels;read(*,*) maps(j)%weights
      end do
      if(m==1) single=maps(1)
    end if
    if(ni>=0) then
      allocate(indices(ni));read(*,*) indices
    end if
    choice='none';if(mode==1) choice='balanced'
    select case(op)
    case(0)
      w=compute_class_weight(classes,y(:,1),choice,single)
    case(1)
      w=compute_sample_weight(y(:,1),choice,single,indices)
    case(2)
      w=compute_sample_weight(y,choice,maps,indices)
    case(3)
      w=compute_sample_weight(y(:,1)/=0,choice,single,indices)
    case(4)
      w=compute_sample_weight(y/=0,choice,maps,indices)
    case(5,6)
      if(op==5) then
        sparse=csr_from_dense(real(y,dp))
      else
        allocate(pointers(n+1),columns(2*n*q),values(2*n*q));pos=1;pointers(1)=1
        do i=1,n
          do j=q,1,-1
            columns(pos:pos+1)=j;values(pos:pos+1)=real(y(i,j),dp)*.5_dp;pos=pos+2
          end do
          pointers(i+1)=pos
        end do
        sparse=csr_from_arrays(n,q,pointers,columns,values);deallocate(pointers,columns,values)
      end if
      w=compute_sample_weight(sparse,choice,maps,indices)
    case(7)
      w=compute_class_weight(classes/=0,y(:,1)/=0,choice,single)
    end select
    write(*,*) size(w)
    write(*,'(*(ES26.17E3,1X))') w
    deallocate(y,classes)
    if(allocated(maps)) deallocate(maps)
    if(allocated(single)) deallocate(single)
    if(allocated(indices)) deallocate(indices)
  end do
end program
