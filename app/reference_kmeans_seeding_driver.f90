program reference_kmeans_seeding_driver
  use sklearn_kmeans_seeding, only: kmeans_plusplus
  use sklearn_common, only: dp
  use sklearn_sparse, only: csr_matrix,csr_from_dense,csr_from_arrays
  implicit none
  integer :: cases,c,n,p,k,seed,trials,rep,nw,nn,i,j,pos,nseed
  integer, allocatable :: ids(:),ptr(:),cols(:),trial_count,before(:),after(:)
  real(dp), allocatable :: x(:,:),original(:,:),centers(:,:),weights(:),norms(:),values(:)
  type(csr_matrix) :: sparse,saved
  logical :: preserved
  call random_seed(size=nseed);allocate(before(nseed),after(nseed))
  read(*,*) cases
  do c=1,cases
    read(*,*) n,p,k,seed,trials,rep,nw,nn
    allocate(x(n,p));read(*,*) x;original=x
    if(nw>=0) then
      allocate(weights(nw));read(*,*) weights
    end if
    if(nn>=0) then
      allocate(norms(nn));read(*,*) norms
    end if
    if(trials/=-1) trial_count=trials
    call random_seed(get=before)
    if(rep==0) then
      call kmeans_plusplus(x,k,centers,ids,weights,norms,seed,trial_count)
      preserved=all(x==original)
    else
      if(rep==1) then
        sparse=csr_from_dense(x)
      else
        allocate(ptr(n+1),cols(2*n*p),values(2*n*p));ptr(1)=1;pos=1
        do i=1,n
          do j=p,1,-1
            cols(pos:pos+1)=j;values(pos:pos+1)=x(i,j)*.5_dp;pos=pos+2
          end do
          ptr(i+1)=pos
        end do
        sparse=csr_from_arrays(n,p,ptr,cols,values);deallocate(ptr,cols,values)
      end if
      saved=sparse
      call kmeans_plusplus(sparse,k,centers,ids,weights,norms,seed,trial_count)
      preserved=all(sparse%data==saved%data).and.all(sparse%indices==saved%indices).and. &
        all(sparse%indptr==saved%indptr)
    end if
    call random_seed(get=after);preserved=preserved.and.all(before==after)
    write(*,*) k,p
    write(*,*) ids
    write(*,'(*(ES26.17E3,1X))') centers
    write(*,*) merge(1,0,preserved)
    deallocate(x)
    if(allocated(weights)) deallocate(weights)
    if(allocated(norms)) deallocate(norms)
    if(allocated(trial_count)) deallocate(trial_count)
  end do
end program
